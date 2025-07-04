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
                       mask_id: int, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply forward masking process."""
        pass


class DiffusionMaskingStrategy(MaskingStrategy):
    """Masking strategy for diffusion language models with caching."""
    
    def __init__(self, p_mask_prompt: float = 0.3, cache_size: int = 1000):
        self.p_mask_prompt = p_mask_prompt
        self.cache_size = cache_size
        self._mask_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def forward_process(self, batch: torch.Tensor, prompt_index: torch.Tensor, 
                       mask_id: int, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply forward masking process with caching."""
        if seed is not None:
            set_seed(seed)
        
        b, l = batch.shape
        device = batch.device
        
        # Create cache key based on batch shape and seed
        cache_key = (b, l, seed, tuple(prompt_index.cpu().numpy()) if prompt_index.numel() < 100 else None)
        
        # Check cache first
        if cache_key in self._mask_cache:
            self._cache_hits += 1
            cached_mask, cached_p_mask = self._mask_cache[cache_key]
            return self._apply_cached_mask(batch, cached_mask, mask_id), cached_p_mask
        
        self._cache_misses += 1
        
        # Generate new mask
        t_p = torch.full((b,), self.p_mask_prompt, device=device)
        random_matrix = torch.rand((b, l), device=device)
        
        # Create mask based on prompt/completion distinction
        is_mask_prompt = prompt_index.unsqueeze(0) & (random_matrix < t_p.unsqueeze(1))
        is_mask_completion = ~prompt_index.unsqueeze(0).expand(b, -1)
        is_mask = is_mask_prompt | is_mask_completion
        
        # Build probability mask
        p_mask = torch.where(
            prompt_index.unsqueeze(0).expand(b, -1),
            t_p.unsqueeze(1),
            torch.ones_like(t_p).unsqueeze(1)
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
                       mask_id: int, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply forward masking using precomputed patterns."""
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
                       mask_id: int, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply adaptive forward masking."""
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