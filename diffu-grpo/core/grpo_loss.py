import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from .memory_manager import MemoryManager, batch_tensor_cleanup
from .masking_strategy import MaskingStrategy
import logging

logger = logging.getLogger(__name__)


class GRPOLoss:
    """Optimized GRPO loss computation with memory management."""
    
    def __init__(self, masking_strategy: MaskingStrategy, 
                 memory_manager: Optional[MemoryManager] = None,
                 epsilon: float = 0.2, beta: float = 0.04,
                 enable_mixed_precision: bool = True):
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
                ratio, clipped_ratio, policy_loss1, policy_loss2, policy_loss, kl_loss
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
        flat_logits = completion_logits.view(-1, completion_logits.size(-1))
        flat_targets = completion_targets.view(-1)
        
        # Use F.cross_entropy with reduction='none' for efficiency
        loss = F.cross_entropy(flat_logits, flat_targets, reduction='none')
        completion_logps = -loss.view(num_iterations * batch_size, logits_to_keep)
        
        # Reshape to final form
        per_token_logps = completion_logps.view(num_iterations, batch_size, logits_to_keep)
        
        # Clean up intermediate tensors
        batch_tensor_cleanup(
            batch_masked, batch_original, logits, completion_logits, 
            completion_targets, flat_logits, flat_targets, loss, completion_logps
        )
        
        return per_token_logps.to(torch.float32)
    
    def _get_old_logps(self, inputs: Dict[str, torch.Tensor], 
                      iteration_idx: int, current_logps: torch.Tensor) -> torch.Tensor:
        """Get old log probabilities for PPO-style updates."""
        
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