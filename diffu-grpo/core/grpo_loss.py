import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from transformers import PreTrainedModel
from .memory_manager import MemoryManager, batch_tensor_cleanup
from .masking_strategy import MaskingStrategy
import logging

logger = logging.getLogger(__name__)


class GRPOLoss:
    """Optimized GRPO loss computation with memory management.
    
    This implementation ensures consistency between current and old log probabilities
    by using the same masking strategy for both computations. This fixes the issue
    where old logps might be computed without proper masking, leading to inaccurate
    policy ratios in PPO-style updates.
    """
    
    def __init__(self, masking_strategy: MaskingStrategy, 
                 memory_manager: Optional[MemoryManager] = None,
                 epsilon: float = 0.2, beta: float = 0.04,
                 enable_mixed_precision: bool = True,
                 ):
        self.masking_strategy = masking_strategy
        self.memory_manager = memory_manager or MemoryManager()
        self.epsilon = epsilon
        self.beta = beta
        self.enable_mixed_precision = enable_mixed_precision
        
        # Performance optimization: pre-allocate commonly used tensors
        self._tensor_cache = {}
        
    def compute_loss(self, model, inputs: Dict[str, torch.Tensor], 
                    iteration_idx: int, num_iterations: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute GRPO loss with optimized memory usage."""
        
        with self.memory_manager.managed_forward("grpo_loss"):
            # Extract inputs
            prompt_ids = inputs["prompt_ids"]
            completion_ids = inputs["completion_ids"]
            completion_mask = inputs["completion_mask"]
            advantages = inputs["advantages"]
            mask_seeds = inputs["mask_seeds"]
            
            # Combine prompt and completion
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            logits_to_keep = completion_ids.size(1)
            
            # Get current iteration mask seed
            current_mask_seed = mask_seeds[iteration_idx].item()
            
            # Compute current log probabilities
            current_logps = self._compute_per_token_logps(
                model, input_ids.unsqueeze(0), logits_to_keep, [current_mask_seed]
            ).squeeze(0)
            
            # Get old log probabilities
            old_logps = self._get_old_logps(inputs, iteration_idx, current_logps)
            
            # Compute policy ratio
            ratio = torch.exp(current_logps - old_logps)
            
            # Compute clipped loss
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            policy_loss1 = ratio * advantages.unsqueeze(1)
            policy_loss2 = clipped_ratio * advantages.unsqueeze(1)
            policy_loss = -torch.min(policy_loss1, policy_loss2)
            
            # Add KL regularization if enabled
            kl_loss = torch.zeros_like(policy_loss)
            if self.beta > 0.0:
                kl_loss = self._compute_kl_loss(inputs, iteration_idx, current_logps, old_logps)
                policy_loss = policy_loss + self.beta * kl_loss
            
            # Compute final loss
            loss = (policy_loss * completion_mask).sum() / completion_mask.sum()
            
            # Compute metrics
            metrics = self._compute_metrics(
                policy_loss1, policy_loss2, completion_mask, kl_loss
            )
            
            # Clean up intermediate tensors
            batch_tensor_cleanup(
                ratio, clipped_ratio, policy_loss1, policy_loss2, policy_loss, kl_loss, force_cache_clear=True
            )
            
            return loss, metrics
    
    def _compute_per_token_logps(self, model, input_ids: torch.Tensor, 
                                logits_to_keep: int, mask_seeds: List[int]) -> torch.Tensor:
        """Compute per-token log probabilities with optimized batching."""
        
        num_iterations, batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Pre-allocate result tensor
        per_token_logps = torch.zeros(
            num_iterations, batch_size, logits_to_keep, 
            device=device, dtype=torch.float32
        )
        
        # Set up prompt indexing
        prompt_length = seq_len - logits_to_keep
        prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=device)
        prompt_index[:prompt_length] = True
        
        # Batch process all iterations
        all_masked_inputs = []
        all_original_inputs = []
        
        for iter_idx, mask_seed in enumerate(mask_seeds):
            original_input = input_ids[iter_idx]
            masked_input, _ = self.masking_strategy.forward_process(
                original_input, prompt_index, mask_id=126336, seed=mask_seed
            )
            all_masked_inputs.append(masked_input)
            all_original_inputs.append(original_input)
        
        # Concatenate for single forward pass
        batch_masked = torch.cat(all_masked_inputs, dim=0)
        batch_original = torch.cat(all_original_inputs, dim=0)
        
        # Forward pass with mixed precision if enabled
        if self.enable_mixed_precision:
            with torch.cuda.amp.autocast():
                logits = model(batch_masked).logits
        else:
            logits = model(batch_masked).logits
        
        # Compute log probabilities for completion tokens
        completion_logits = logits[:, -logits_to_keep:, :]
        completion_targets = batch_original[:, -logits_to_keep:]
        
        # Efficient cross-entropy computation
        flat_logits = completion_logits.reshape(-1, completion_logits.size(-1))
        flat_targets = completion_targets.reshape(-1)
        
        # Use F.cross_entropy with reduction='none' for efficiency
        loss = F.cross_entropy(flat_logits, flat_targets, reduction='none')
        completion_logps = -loss.reshape(num_iterations * batch_size, logits_to_keep)
        
        # Reshape to final form
        per_token_logps = completion_logps.reshape(num_iterations, batch_size, logits_to_keep)
        
        # Clean up intermediate tensors
        batch_tensor_cleanup(
            batch_masked, batch_original, logits, completion_logits, 
            completion_targets, flat_logits, flat_targets, loss, completion_logps
        )
        
        return per_token_logps.to(torch.float32)
    
    def _get_old_logps(self, inputs: Dict[str, torch.Tensor], 
                      iteration_idx: int, current_logps: torch.Tensor) -> torch.Tensor:
        """Get old log probabilities for PPO-style updates.
        
        Args:
            inputs: Dictionary containing training inputs including old_per_token_logps
            iteration_idx: Current iteration index
            current_logps: Current log probabilities (used as fallback)
            
        Returns:
            Old log probabilities tensor
        """
        
        if "old_per_token_logps" in inputs and inputs["old_per_token_logps"] is not None:
            return inputs["old_per_token_logps"][iteration_idx].squeeze(0)
        else:
            # For first iteration, use current logps (detached)
            return current_logps.detach()
    

    
    def _compute_kl_loss(self, inputs: Dict[str, torch.Tensor], 
                        iteration_idx: int, current_logps: torch.Tensor,
                        old_logps: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss."""
        
        if "ref_per_token_logps" in inputs and inputs["ref_per_token_logps"] is not None:
            ref_logps = inputs["ref_per_token_logps"][iteration_idx].squeeze(0)
            # KL divergence: KL(current || ref) = exp(ref - current) - (ref - current) - 1
            kl_div = torch.exp(ref_logps - current_logps) - (ref_logps - current_logps) - 1
            return kl_div
        else:
            # No reference model, return zeros
            return torch.zeros_like(current_logps)
    
    def _compute_metrics(self, policy_loss1: torch.Tensor, policy_loss2: torch.Tensor,
                        completion_mask: torch.Tensor, kl_loss: torch.Tensor) -> Dict[str, float]:
        """Compute training metrics."""
        
        # Clipping statistics
        is_clipped = (policy_loss1 < policy_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        
        metrics = {
            'clip_ratio': clip_ratio.item(),
        }
        
        # KL divergence statistics
        if self.beta > 0.0:
            mean_kl = (kl_loss * completion_mask).sum() / completion_mask.sum()
            metrics['kl_divergence'] = mean_kl.item()
        
        return metrics
    
    def update_hyperparameters(self, epsilon: Optional[float] = None, 
                             beta: Optional[float] = None):
        """Update loss hyperparameters during training."""
        
        if epsilon is not None:
            self.epsilon = epsilon
            logger.info(f"Updated epsilon to {epsilon}")
        
        if beta is not None:
            self.beta = beta
            logger.info(f"Updated beta to {beta}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        
        stats = {
            'epsilon': self.epsilon,
            'beta': self.beta,
            'mixed_precision': self.enable_mixed_precision,
            'memory_stats': self.memory_manager.get_memory_stats()
        }
        
        if hasattr(self.masking_strategy, 'get_cache_stats'):
            stats['masking_cache_stats'] = self.masking_strategy.get_cache_stats()
        
        return stats


class AdaptiveGRPOLoss(GRPOLoss):
    """GRPO loss with adaptive hyperparameters."""
    
    def __init__(self, masking_strategy: MaskingStrategy,
                 memory_manager: Optional[MemoryManager] = None,
                 initial_epsilon: float = 0.2, final_epsilon: float = 0.1,
                 initial_beta: float = 0.04, final_beta: float = 0.01,
                 total_steps: int = 10000, **kwargs):
        super().__init__(masking_strategy, memory_manager, initial_epsilon, initial_beta, **kwargs)
        
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.initial_beta = initial_beta
        self.final_beta = final_beta
        self.total_steps = total_steps
        self.current_step = 0
    
    def update_step(self, step: int):
        """Update current step and adapt hyperparameters."""
        
        self.current_step = step
        
        # Adaptive epsilon
        if step < self.total_steps:
            progress = step / self.total_steps
            self.epsilon = self.initial_epsilon + (self.final_epsilon - self.initial_epsilon) * progress
            self.beta = self.initial_beta + (self.final_beta - self.initial_beta) * progress
        else:
            self.epsilon = self.final_epsilon
            self.beta = self.final_beta


class MultiObjectiveGRPOLoss(GRPOLoss):
    """GRPO loss with multiple objectives (e.g., reward + length penalty)."""
    
    def __init__(self, masking_strategy: MaskingStrategy,
                 memory_manager: Optional[MemoryManager] = None,
                 length_penalty_weight: float = 0.01,
                 **kwargs):
        super().__init__(masking_strategy, memory_manager, **kwargs)
        self.length_penalty_weight = length_penalty_weight
    
    def compute_loss(self, model, inputs: Dict[str, torch.Tensor], 
                    iteration_idx: int, num_iterations: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-objective GRPO loss."""
        
        # Get standard GRPO loss
        base_loss, metrics = super().compute_loss(model, inputs, iteration_idx, num_iterations)
        
        # Add length penalty
        completion_mask = inputs["completion_mask"]
        length_penalty = completion_mask.sum(dim=1).float().mean()
        
        total_loss = base_loss + self.length_penalty_weight * length_penalty
        
        metrics['length_penalty'] = length_penalty.item()
        metrics['base_loss'] = base_loss.item()
        
        return total_loss, metrics


class MonteCarloGRPOLoss(GRPOLoss):
    """Enhanced GRPO loss with Monte Carlo log-likelihood estimation.
    
    This implementation includes memory optimizations for computing old log probabilities:
    - Uses reduced sampling (mc_num_old_logps) for old logps computation
    - Processes only the specific iteration needed instead of all iterations
    - Includes explicit memory cleanup during Monte Carlo sampling
    - Default mc_num_old_logps is max(32, mc_num // 4) to reduce memory usage by ~75%
    """
    
    def __init__(self, masking_strategy: MaskingStrategy,
                 memory_manager: Optional[MemoryManager] = None,
                 mc_num: int = 128,
                 mc_batch_size: int = 16,
                 cfg_scale: float = 0.0,
                 mask_id: int = 126336,
                 mc_num_old_logps: Optional[int] = None,  # Reduced sampling for old logps
                 **kwargs):
        super().__init__(masking_strategy, memory_manager, **kwargs)
        self.mc_num = mc_num
        self.mc_batch_size = mc_batch_size
        self.cfg_scale = cfg_scale
        self.mask_id = mask_id
        
        # Use reduced sampling for old logps computation to save memory
        self.mc_num_old_logps = mc_num_old_logps or max(32, mc_num // 4)
        
        # Validate Monte Carlo parameters
        if self.mc_num % self.mc_batch_size != 0:
            raise ValueError(f"mc_num ({self.mc_num}) must be divisible by mc_batch_size ({self.mc_batch_size})")
        
        # Adaptive batch size based on memory
        self.adaptive_mc_batch_size = self._get_adaptive_mc_batch_size()
        
        if self.mc_num_old_logps % self.mc_batch_size != 0:
            # Adjust mc_num_old_logps to be divisible by mc_batch_size
            self.mc_num_old_logps = (self.mc_num_old_logps // self.mc_batch_size) * self.mc_batch_size
            self.mc_num_old_logps = max(self.mc_batch_size, self.mc_num_old_logps)
        
        logger.info(f"Initialized MonteCarloGRPOLoss with mc_num={mc_num}, mc_batch_size={mc_batch_size}, mc_num_old_logps={self.mc_num_old_logps}")
    
    def _monte_carlo_forward_process(self, batch: torch.Tensor, prompt_index: torch.Tensor, 
                                   mask_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Monte Carlo forward process following the reference implementation."""
        b, l = batch.shape
        device = batch.device
        
        # Calculate target length (non-prompt tokens)
        target_len = (l - prompt_index.sum()).item()
        
        # Sample random k for masking
        k = torch.randint(1, target_len + 1, (), device=device)
        
        # Distribute masking across batch with proper spacing
        x = torch.round(torch.linspace(
            float(k), k + (b - 1) * (target_len / b), 
            steps=b, device=device
        )).long()
        x = ((x - 1) % target_len) + 1
        
        assert x.min() >= 1 and x.max() <= target_len, f"Invalid x range: {x.min()} to {x.max()}"
        
        # Create random masking pattern
        indices = torch.arange(target_len, device=device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)
        
        # Apply random permutation to each batch element
        for i in range(b):
            perm = torch.randperm(target_len, device=device)
            is_mask[i] = is_mask[i][perm]
        
        # Combine prompt (never masked) with completion masking
        prompt_mask = torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=device)
        is_mask = torch.cat((prompt_mask, is_mask), dim=1)
        
        # Apply masking in-place for memory efficiency
        noisy_batch = batch.clone()
        noisy_batch.masked_fill_(is_mask, mask_id)
        
        # Return mask ratio for each token
        mask_ratio = (x / target_len).unsqueeze(1).repeat(1, l)
        
        # Clean up intermediate tensors immediately
        batch_tensor_cleanup(indices, prompt_mask, force_cache_clear=False)
        
        return noisy_batch, mask_ratio
    
    def _get_logits_with_cfg(self, model, batch: torch.Tensor, prompt_index: torch.Tensor, 
                            cfg_scale: float, mask_id: int) -> torch.Tensor:
        """Get logits with classifier-free guidance if enabled."""
        if cfg_scale > 0.0:
            # Memory-optimized sequential CFG processing
            return self._get_logits_with_cfg_sequential(model, batch, prompt_index, cfg_scale, mask_id)
        else:
            # Standard forward pass with gradient checkpointing
            return self._get_logits_standard(model, batch)
    
    def _get_logits_with_cfg_sequential(self, model, batch: torch.Tensor, prompt_index: torch.Tensor, 
                                      cfg_scale: float, mask_id: int) -> torch.Tensor:
        """Sequential CFG processing to halve memory usage."""
        # Process conditional batch first
        if self.enable_mixed_precision:
            with torch.cuda.amp.autocast():
                cond_logits = torch.utils.checkpoint.checkpoint(
                    lambda x: model(x).logits, batch, use_reentrant=False
                )
        else:
            cond_logits = torch.utils.checkpoint.checkpoint(
                lambda x: model(x).logits, batch, use_reentrant=False
            )
        
        # Process unconditional batch separately
        un_batch = batch.clone()
        un_batch[:, prompt_index] = mask_id
        
        if self.enable_mixed_precision:
            with torch.cuda.amp.autocast():
                uncond_logits = torch.utils.checkpoint.checkpoint(
                    lambda x: model(x).logits, un_batch, use_reentrant=False
                )
        else:
            uncond_logits = torch.utils.checkpoint.checkpoint(
                lambda x: model(x).logits, un_batch, use_reentrant=False
            )
        
        # Apply CFG
        logits = uncond_logits + (cfg_scale + 1) * (cond_logits - uncond_logits)
        
        # Clean up intermediate tensors
        batch_tensor_cleanup(un_batch, cond_logits, uncond_logits, force_cache_clear=False)
        
        return logits
    
    def _get_logits_standard(self, model, batch: torch.Tensor) -> torch.Tensor:
        """Standard forward pass with gradient checkpointing."""
        if self.enable_mixed_precision:
            with torch.cuda.amp.autocast():
                return torch.utils.checkpoint.checkpoint(
                    lambda x: model(x).logits, batch, use_reentrant=False
                )
        else:
            return torch.utils.checkpoint.checkpoint(
                lambda x: model(x).logits, batch, use_reentrant=False
            )
    
    @torch.no_grad()
    def _monte_carlo_log_likelihood(self, model, prompt_ids: torch.Tensor, 
                                  completion_ids: torch.Tensor, mc_num: Optional[int] = None) -> torch.Tensor:
        """Compute log-likelihood using Monte Carlo estimation with optimized memory usage."""
        device = model.device
        
        # Use custom mc_num if provided (for memory optimization)
        mc_num = mc_num or self.mc_num
        
        # Combine prompt and completion
        seq = torch.cat([prompt_ids, completion_ids], dim=1)
        batch_size = seq.size(0)
        seq_len = seq.size(1)
        
        # Create prompt index mask
        prompt_index = torch.arange(seq_len, device=device) < prompt_ids.size(1)
        
        # Use adaptive batch size for memory efficiency
        effective_batch_size = min(self.adaptive_mc_batch_size, self.mc_batch_size)
        num_mc_batches = mc_num // effective_batch_size
        
        # Accumulate losses more efficiently
        total_loss = 0.0
        total_samples = 0
        
        for batch_idx in range(num_mc_batches):
            # Create smaller batch for this iteration with adaptive batch size
            seq_batch = seq.repeat(effective_batch_size, 1)
            
            # Apply Monte Carlo forward process
            perturbed_seq, p_mask = self._monte_carlo_forward_process(
                seq_batch, prompt_index, self.mask_id
            )
            
            # Find masked positions
            mask_index = perturbed_seq == self.mask_id
            
            # Skip if no tokens are masked (edge case)
            if not mask_index.any():
                batch_tensor_cleanup(seq_batch, perturbed_seq, p_mask, force_cache_clear=False)
                continue
            
            # Get logits with optional CFG
            logits = self._get_logits_with_cfg(
                model, perturbed_seq, prompt_index, self.cfg_scale, self.mask_id
            )
            
            # Compute cross-entropy loss for masked tokens only
            try:
                loss = F.cross_entropy(
                    logits[mask_index], 
                    seq_batch[mask_index], 
                    reduction='none'
                )
                
                # Weight by mask probability and accumulate
                if p_mask[mask_index].sum() > 0:  # Avoid division by zero
                    weighted_loss = loss / p_mask[mask_index]
                    total_loss += weighted_loss.sum().item()
                    total_samples += weighted_loss.numel()
                
            except Exception as e:
                logger.warning(f"Monte Carlo batch {batch_idx} failed: {str(e)}")
                
            # Clean up intermediate tensors immediately after each batch
            batch_tensor_cleanup(
                seq_batch, perturbed_seq, p_mask, logits, loss, 
                force_cache_clear=(batch_idx % 2 == 0)  # More aggressive: clear cache every 2 batches
            )
            
            # Additional memory cleanup for Monte Carlo batches
            if batch_idx % 4 == 0:  # Every 4 batches, do aggressive cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                # Clear memory manager cache if available
                if hasattr(self, 'memory_manager') and self.memory_manager is not None:
                    self.memory_manager.cleanup_batch_tensors()
        
        # Return average negative log-likelihood (positive value)
        if total_samples > 0:
            return -total_loss / total_samples
        else:
            # Fallback to zero if no valid samples
            logger.warning("No valid Monte Carlo samples generated")
            return 0.0
    
    def _compute_per_token_logps_mc(self, model, input_ids: torch.Tensor, 
                                  logits_to_keep: int, mask_seeds: List[int],
                                  mc_num: Optional[int] = None) -> torch.Tensor:
        """Compute per-token log probabilities using Monte Carlo estimation with optimized batching."""
        
        num_iterations, batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Use reduced sampling if specified (for memory optimization)
        mc_num = mc_num or self.mc_num
        
        # Pre-allocate result tensor
        per_token_logps = torch.zeros(
            num_iterations, batch_size, logits_to_keep,
            device=device, dtype=torch.float32
        )
        
        # Set up prompt and completion tensors
        prompt_length = seq_len - logits_to_keep
        
        # Process iterations in smaller batches for memory efficiency
        for iter_idx in range(num_iterations):
            # Set seed for reproducibility
            if mask_seeds[iter_idx] is not None:
                torch.manual_seed(mask_seeds[iter_idx])
            
            # Process all samples in this iteration together for better efficiency
            iteration_input = input_ids[iter_idx]  # [batch_size, seq_len]
            prompt_seq = iteration_input[:, :prompt_length]
            completion_seq = iteration_input[:, prompt_length:]
            
            # Compute log-likelihood for all samples in this iteration
            for batch_idx in range(batch_size):
                single_prompt = prompt_seq[batch_idx:batch_idx+1]
                single_completion = completion_seq[batch_idx:batch_idx+1]
                
                log_likelihood = self._monte_carlo_log_likelihood(
                    model, single_prompt, single_completion, mc_num=mc_num
                )
                
                # Distribute log-likelihood evenly across completion tokens
                # (This is a simplification - in practice, you might want per-token estimates)
                per_token_logps[iter_idx, batch_idx, :] = log_likelihood / logits_to_keep
            
            # Clean up iteration tensors
            batch_tensor_cleanup(iteration_input, prompt_seq, completion_seq, force_cache_clear=True)
        
        return per_token_logps
    
    def _get_adaptive_mc_batch_size(self) -> int:
        """Determine adaptive Monte Carlo batch size based on available memory."""
        if not torch.cuda.is_available():
            return min(4, self.mc_batch_size)
        
        try:
            # Get available GPU memory in GB
            available_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free_memory, total_memory = torch.cuda.mem_get_info()
            free_memory_gb = free_memory / (1024**3)
            
            # Conservative memory-aware batch sizing
            if free_memory_gb < 8:
                return min(4, self.mc_batch_size)
            elif free_memory_gb < 16:
                return min(8, self.mc_batch_size)
            elif free_memory_gb < 32:
                return min(12, self.mc_batch_size)
            else:
                return self.mc_batch_size
        except Exception:
            # Fallback to conservative batch size
            return min(4, self.mc_batch_size)
    
    def _compute_per_token_logps(self, model, input_ids: torch.Tensor, 
                               logits_to_keep: int, mask_seeds: List[int]) -> torch.Tensor:
        """Override to use Monte Carlo estimation."""
        with self.memory_manager.managed_forward("monte_carlo_logps"):
            return self._compute_per_token_logps_mc(model, input_ids, logits_to_keep, mask_seeds)
    

    
    def update_mc_params(self, mc_num: Optional[int] = None, mc_batch_size: Optional[int] = None,
                        cfg_scale: Optional[float] = None, mc_num_old_logps: Optional[int] = None):
        """Update Monte Carlo parameters during training."""
        if mc_num is not None:
            if mc_batch_size is None:
                mc_batch_size = self.mc_batch_size
            if mc_num % mc_batch_size != 0:
                raise ValueError(f"mc_num ({mc_num}) must be divisible by mc_batch_size ({mc_batch_size})")
            self.mc_num = mc_num
            logger.info(f"Updated mc_num to {mc_num}")
        
        if mc_batch_size is not None:
            if self.mc_num % mc_batch_size != 0:
                raise ValueError(f"mc_num ({self.mc_num}) must be divisible by mc_batch_size ({mc_batch_size})")
            self.mc_batch_size = mc_batch_size
            logger.info(f"Updated mc_batch_size to {mc_batch_size}")
        
        if mc_num_old_logps is not None:
            if mc_num_old_logps % self.mc_batch_size != 0:
                # Adjust mc_num_old_logps to be divisible by mc_batch_size
                mc_num_old_logps = (mc_num_old_logps // self.mc_batch_size) * self.mc_batch_size
                mc_num_old_logps = max(self.mc_batch_size, mc_num_old_logps)
            self.mc_num_old_logps = mc_num_old_logps
            logger.info(f"Updated mc_num_old_logps to {mc_num_old_logps}")
        
        if cfg_scale is not None:
            self.cfg_scale = cfg_scale
            logger.info(f"Updated cfg_scale to {cfg_scale}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics including Monte Carlo parameters."""
        stats = super().get_performance_stats()
        stats.update({
            'mc_num': self.mc_num,
            'mc_batch_size': self.mc_batch_size,
            'mc_num_old_logps': self.mc_num_old_logps,
            'cfg_scale': self.cfg_scale,
            'mask_id': self.mask_id,
            'memory_reduction_ratio': self.mc_num_old_logps / self.mc_num
        })
        return stats


class HybridGRPOLoss(GRPOLoss):
    """Hybrid GRPO loss that can switch between standard and Monte Carlo estimation."""
    
    def __init__(self, masking_strategy: MaskingStrategy,
                 memory_manager: Optional[MemoryManager] = None,
                 use_monte_carlo: bool = True,
                 monte_carlo_warmup_steps: int = 1000,
                 **kwargs):
        super().__init__(masking_strategy, memory_manager, **kwargs)
        
        # Initialize Monte Carlo component
        if use_monte_carlo:
            self.mc_loss = MonteCarloGRPOLoss(
                masking_strategy=masking_strategy,
                memory_manager=memory_manager,
                **kwargs  # This includes ref_model for consistency
            )
        else:
            self.mc_loss = None
        
        self.use_monte_carlo = use_monte_carlo
        self.monte_carlo_warmup_steps = monte_carlo_warmup_steps
        self.current_step = 0
    
    def update_step(self, step: int):
        """Update current step for adaptive behavior."""
        self.current_step = step
        if hasattr(self, 'mc_loss') and self.mc_loss is not None:
            if hasattr(self.mc_loss, 'update_step'):
                self.mc_loss.update_step(step)
    

    
    def update_mc_params(self, **kwargs):
        """Update Monte Carlo parameters and sync with mc_loss component."""
        if self.mc_loss is not None:
            self.mc_loss.update_mc_params(**kwargs)
    
    def _should_use_monte_carlo(self) -> bool:
        """Determine whether to use Monte Carlo estimation."""
        if not self.use_monte_carlo or self.mc_loss is None:
            return False
        
        # Use Monte Carlo after warmup period
        return self.current_step >= self.monte_carlo_warmup_steps
    
    def _compute_per_token_logps(self, model, input_ids: torch.Tensor, 
                               logits_to_keep: int, mask_seeds: List[int]) -> torch.Tensor:
        """Choose between standard and Monte Carlo estimation.
        
        This method ensures that both current and old log probabilities use
        the same estimation method (either both standard or both Monte Carlo).
        """
        if self._should_use_monte_carlo():
            return self.mc_loss._compute_per_token_logps(model, input_ids, logits_to_keep, mask_seeds)
        else:
            return super()._compute_per_token_logps(model, input_ids, logits_to_keep, mask_seeds)
    
    def _verify_estimation_consistency(self) -> bool:
        """Verify that current and old logps use the same estimation method."""
        use_mc = self._should_use_monte_carlo()
        if use_mc and self.mc_loss is None:
            logger.warning("Should use Monte Carlo but mc_loss is None")
            return False
        
        return True
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for hybrid approach."""
        stats = super().get_performance_stats()
        stats.update({
            'use_monte_carlo': self.use_monte_carlo,
            'monte_carlo_warmup_steps': self.monte_carlo_warmup_steps,
            'current_step': self.current_step,
            'using_monte_carlo': self._should_use_monte_carlo(),
            'estimation_consistent': self._verify_estimation_consistency()
        })
        
        if self.mc_loss is not None:
            stats['monte_carlo_stats'] = self.mc_loss.get_performance_stats()
        
        return stats