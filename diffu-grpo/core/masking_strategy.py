import torch
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
from accelerate.utils import set_seed
import logging

logger = logging.getLogger(__name__)


class MaskingStrategy(ABC):
    """Abstract base class for masking strategies."""
    
    @abstractmethod
    def forward_process(self, batch: torch.Tensor, prompt_index: torch.Tensor, 
                       mask_id: int, seed: Optional[int] = None, timestep: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply forward masking process."""
        pass


class DiffusionMaskingStrategy(MaskingStrategy):
    """Masking strategy for diffusion language models with caching and alpha scheduling."""
    
    def __init__(self, p_mask_prompt: float = 0.3, cache_size: int = 1000,
                 max_timesteps: int = 128, alpha_schedule: str = "cosine",
                 alpha_min: float = 0.01, alpha_max: float = 0.99):
        self.p_mask_prompt = p_mask_prompt
        self.cache_size = cache_size
        self._mask_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Alpha schedule parameters for consistency with TrajectoryNELBOLoss
        self.max_timesteps = max_timesteps
        self.alpha_schedule = alpha_schedule
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        
        # Initialize alpha schedule
        self.alpha_t = self._create_alpha_schedule()
        
        logger.info(f"DiffusionMaskingStrategy initialized with {alpha_schedule} schedule, {max_timesteps} timesteps")
    
    def _create_alpha_schedule(self) -> torch.Tensor:
        """Create noise schedule α_t consistent with TrajectoryNELBOLoss."""
        t = torch.linspace(0, 1, self.max_timesteps + 1)
        
        if self.alpha_schedule == "cosine":
            # Cosine schedule: smoother transitions
            alpha_t = torch.cos(t * torch.pi / 2) ** 2
            alpha_t = self.alpha_min + (self.alpha_max - self.alpha_min) * alpha_t
        elif self.alpha_schedule == "linear":
            # Linear schedule
            alpha_t = self.alpha_max - (self.alpha_max - self.alpha_min) * t
        elif self.alpha_schedule == "exponential":
            # Exponential decay
            alpha_t = self.alpha_max * torch.exp(-3 * t)
            alpha_t = torch.clamp(alpha_t, self.alpha_min, self.alpha_max)
        else:
            raise ValueError(f"Unknown alpha schedule: {self.alpha_schedule}")
        
        # Ensure monotonic decrease and proper bounds
        alpha_t = torch.maximum(alpha_t, torch.tensor(self.alpha_min))
        alpha_t = torch.minimum(alpha_t, torch.tensor(self.alpha_max))
        
        return alpha_t
    
    def _get_masking_probability_from_alpha(self, timestep: int, device: torch.device) -> float:
        """Convert alpha schedule to masking probability for given timestep."""
        if timestep <= 0:
            return 0.0  # No masking at t=0
        if timestep >= self.max_timesteps:
            return 1.0  # Full masking at max timesteps
        
        # Get alpha value for this timestep
        alpha = self.alpha_t[timestep].item()
        
        # Convert alpha (probability of keeping original) to masking probability
        # For completion tokens: p_mask = 1 - alpha (more noise = more masking)
        # For prompt tokens: use configured p_mask_prompt
        return 1.0 - alpha
    
    def forward_process(self, batch: torch.Tensor, prompt_index: torch.Tensor, 
                       mask_id: int, seed: Optional[int] = None, timestep: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply forward masking process with alpha scheduling and caching."""
        if seed is not None:
            set_seed(seed)
        
        b, l = batch.shape
        device = batch.device
        
        # Determine timestep for alpha scheduling
        if timestep is None:
            # If no timestep provided, sample one randomly (for training variability)
            if seed is not None:
                torch.manual_seed(seed)
            timestep = torch.randint(1, self.max_timesteps + 1, (1,)).item()
        
        # Create cache key based on batch shape, seed, and timestep
        cache_key = (b, l, seed, timestep, tuple(prompt_index.cpu().numpy()) if prompt_index.numel() < 100 else None)
        
        # Check cache first
        if cache_key in self._mask_cache:
            self._cache_hits += 1
            cached_mask, cached_p_mask = self._mask_cache[cache_key]
            return self._apply_cached_mask(batch, cached_mask, mask_id), cached_p_mask
        
        self._cache_misses += 1
        
        # Get masking probabilities based on alpha schedule
        alpha_t = self.alpha_t[timestep].to(device)
        
        # For prompt tokens: use configured p_mask_prompt (manual control)
        # For completion tokens: use alpha schedule (1 - alpha_t = noise level)
        t_p_prompt = torch.full((b,), self.p_mask_prompt, device=device)
        t_p_completion = torch.full((b,), 1.0 - alpha_t.item(), device=device)
        
        # Generate random matrix for masking decisions
        random_matrix = torch.rand((b, l), device=device)
        
        # Create mask based on prompt/completion distinction with alpha scheduling
        prompt_mask_expanded = prompt_index.unsqueeze(0).expand(b, -1)
        completion_mask_expanded = ~prompt_mask_expanded
        
        # Apply different masking probabilities for prompt vs completion
        is_mask_prompt = prompt_mask_expanded & (random_matrix < t_p_prompt.unsqueeze(1))
        is_mask_completion = completion_mask_expanded & (random_matrix < t_p_completion.unsqueeze(1))
        is_mask = is_mask_prompt | is_mask_completion
        
        # Build probability mask (for loss computation consistency)
        p_mask = torch.where(
            prompt_mask_expanded,
            t_p_prompt.unsqueeze(1),
            t_p_completion.unsqueeze(1)
        )
        
        # Cache the mask if cache isn't full
        if len(self._mask_cache) < self.cache_size:
            self._mask_cache[cache_key] = (is_mask.clone(), p_mask.clone())
        
        # Apply mask
        noisy_batch = torch.where(is_mask, mask_id, batch)
        
        return noisy_batch, p_mask
    
    def _apply_cached_mask(self, batch: torch.Tensor, cached_mask: torch.Tensor, mask_id: int) -> torch.Tensor:
        """Apply a cached mask to the batch."""
        return torch.where(cached_mask, mask_id, batch)
    
    def get_cache_stats(self) -> dict:
        """Get cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._mask_cache)
        }
    
    def clear_cache(self):
        """Clear the mask cache."""
        self._mask_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def update_alpha_schedule(self, max_timesteps: int = None, alpha_schedule: str = None,
                            alpha_min: float = None, alpha_max: float = None):
        """Update alpha schedule parameters and regenerate schedule."""
        updated = False
        
        if max_timesteps is not None:
            self.max_timesteps = max_timesteps
            updated = True
            
        if alpha_schedule is not None:
            self.alpha_schedule = alpha_schedule
            updated = True
            
        if alpha_min is not None:
            self.alpha_min = alpha_min
            updated = True
            
        if alpha_max is not None:
            self.alpha_max = alpha_max
            updated = True
        
        if updated:
            self.alpha_t = self._create_alpha_schedule()
            self.clear_cache()  # Clear cache since schedule changed
            logger.info(f"Updated DiffusionMaskingStrategy: {self.alpha_schedule} schedule, {self.max_timesteps} timesteps")
    
    def get_alpha_schedule_params(self) -> dict:
        """Get current alpha schedule parameters for consistency checking."""
        return {
            'max_timesteps': self.max_timesteps,
            'alpha_schedule': self.alpha_schedule,
            'alpha_min': self.alpha_min,
            'alpha_max': self.alpha_max,
            'alpha_t': self.alpha_t.clone()
        }
    
    def sync_with_nelbo_loss(self, nelbo_loss):
        """Synchronize alpha schedule with TrajectoryNELBOLoss for consistency."""
        if hasattr(nelbo_loss, 'max_timesteps'):
            self.update_alpha_schedule(
                max_timesteps=nelbo_loss.max_timesteps,
                alpha_schedule=nelbo_loss.alpha_schedule,
                alpha_min=nelbo_loss.alpha_min,
                alpha_max=nelbo_loss.alpha_max
            )
            logger.info("✓ DiffusionMaskingStrategy synchronized with TrajectoryNELBOLoss")


class OptimizedMaskingStrategy(MaskingStrategy):
    """Optimized masking strategy with pre-computed patterns."""
    
    def __init__(self, p_mask_prompt: float = 0.3, num_precomputed: int = 100):
        self.p_mask_prompt = p_mask_prompt
        self.num_precomputed = num_precomputed
        self._precomputed_patterns = {}
        self._pattern_index = 0
    
    def precompute_patterns(self, batch_size: int, sequence_length: int, device: torch.device):
        """Precompute masking patterns for common configurations."""
        pattern_key = (batch_size, sequence_length)
        if pattern_key in self._precomputed_patterns:
            return
        
        patterns = []
        for i in range(self.num_precomputed):
            with torch.random.fork_rng():
                torch.manual_seed(i)
                pattern = torch.rand(batch_size, sequence_length, device=device) < self.p_mask_prompt
                patterns.append(pattern)
        
        self._precomputed_patterns[pattern_key] = patterns
        logger.info(f"Precomputed {self.num_precomputed} masking patterns for shape {pattern_key}")
    
    def forward_process(self, batch: torch.Tensor, prompt_index: torch.Tensor, 
                       mask_id: int, seed: Optional[int] = None, timestep: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply forward masking using precomputed patterns."""
        # timestep parameter ignored for this strategy
        _ = timestep
        
        b, l = batch.shape
        device = batch.device
        
        # Ensure patterns are precomputed
        self.precompute_patterns(b, l, device)
        
        # Get pattern based on seed or cycle through precomputed
        pattern_key = (b, l)
        if seed is not None:
            pattern_idx = seed % self.num_precomputed
        else:
            pattern_idx = self._pattern_index % self.num_precomputed
            self._pattern_index += 1
        
        prompt_mask_pattern = self._precomputed_patterns[pattern_key][pattern_idx]
        
        # Apply masking logic
        is_mask_prompt = prompt_index.unsqueeze(0) & prompt_mask_pattern
        is_mask_completion = ~prompt_index.unsqueeze(0).expand(b, -1)
        is_mask = is_mask_prompt | is_mask_completion
        
        # Create probability mask
        t_p = torch.full((b,), self.p_mask_prompt, device=device)
        p_mask = torch.where(
            prompt_index.unsqueeze(0).expand(b, -1),
            t_p.unsqueeze(1),
            torch.ones_like(t_p).unsqueeze(1)
        )
        
        # Apply mask
        noisy_batch = torch.where(is_mask, mask_id, batch)
        
        return noisy_batch, p_mask


class AdaptiveMaskingStrategy(MaskingStrategy):
    """Adaptive masking strategy that adjusts based on training progress."""
    
    def __init__(self, initial_p_mask: float = 0.3, final_p_mask: float = 0.1, 
                 total_steps: int = 10000):
        self.initial_p_mask = initial_p_mask
        self.final_p_mask = final_p_mask
        self.total_steps = total_steps
        self.current_step = 0
    
    def update_step(self, step: int):
        """Update the current training step."""
        self.current_step = step
    
    def get_current_p_mask(self) -> float:
        """Get the current masking probability based on training progress."""
        if self.current_step >= self.total_steps:
            return self.final_p_mask
        
        # Linear interpolation
        progress = self.current_step / self.total_steps
        return self.initial_p_mask + (self.final_p_mask - self.initial_p_mask) * progress
    
    def forward_process(self, batch: torch.Tensor, prompt_index: torch.Tensor, 
                       mask_id: int, seed: Optional[int] = None, timestep: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply adaptive forward masking."""
        # timestep parameter ignored for this strategy
        _ = timestep
        
        if seed is not None:
            set_seed(seed)
        
        b, l = batch.shape
        device = batch.device
        
        # Use adaptive masking probability
        current_p_mask = self.get_current_p_mask()
        t_p = torch.full((b,), current_p_mask, device=device)
        
        # Generate mask
        random_matrix = torch.rand((b, l), device=device)
        is_mask_prompt = prompt_index.unsqueeze(0) & (random_matrix < t_p.unsqueeze(1))
        is_mask_completion = ~prompt_index.unsqueeze(0).expand(b, -1)
        is_mask = is_mask_prompt | is_mask_completion
        
        # Build probability mask
        p_mask = torch.where(
            prompt_index.unsqueeze(0).expand(b, -1),
            t_p.unsqueeze(1),
            torch.ones_like(t_p).unsqueeze(1)
        )
        
        # Apply mask
        noisy_batch = torch.where(is_mask, mask_id, batch)
        
        return noisy_batch, p_mask


def create_masking_strategy(strategy_type: str = "diffusion", **kwargs) -> MaskingStrategy:
    """Factory function to create masking strategies."""
    if strategy_type == "diffusion":
        return DiffusionMaskingStrategy(**kwargs)
    elif strategy_type == "optimized":
        return OptimizedMaskingStrategy(**kwargs)
    elif strategy_type == "adaptive":
        return AdaptiveMaskingStrategy(**kwargs)
    else:
        raise ValueError(f"Unknown masking strategy: {strategy_type}")


def sync_masking_with_nelbo(masking_strategy: MaskingStrategy, nelbo_loss) -> None:
    """Synchronize masking strategy with NELBO loss for consistent scheduling."""
    if isinstance(masking_strategy, DiffusionMaskingStrategy) and hasattr(nelbo_loss, 'max_timesteps'):
        masking_strategy.sync_with_nelbo_loss(nelbo_loss)
        logger.info("✓ Masking strategy and NELBO loss schedules synchronized")
    else:
        logger.warning("⚠️ Could not synchronize masking strategy with NELBO loss")