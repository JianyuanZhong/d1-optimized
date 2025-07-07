import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from .memory_manager import MemoryManager, batch_tensor_cleanup
import logging

logger = logging.getLogger(__name__)


class DiffusionGenerator:
    """Optimized diffusion generation with memory management and batching."""
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None, 
                 enable_mixed_precision: bool = True, cache_logits: bool = True):
        self.memory_manager = memory_manager or MemoryManager()
        self.enable_mixed_precision = enable_mixed_precision
        self.cache_logits = cache_logits
        self._logits_cache = {}
        
    def generate(self, model, prompt: torch.Tensor, steps: int = 128, 
                 gen_length: int = 128, block_length: int = 128, 
                 temperature: float = 0.0, cfg_scale: float = 0.0,
                 remasking: str = "low_confidence", mask_id: int = 126336,
                 batch_size: Optional[int] = None) -> torch.Tensor:
        """Optimized generation with better memory management and batching."""
        
        with self.memory_manager.managed_generation(prompt.shape[0], gen_length):
            if batch_size is None:
                return self._generate_single_batch(
                    model, prompt, steps, gen_length, block_length,
                    temperature, cfg_scale, remasking, mask_id
                )
            else:
                return self._generate_multiple_batches(
                    model, prompt, steps, gen_length, block_length,
                    temperature, cfg_scale, remasking, mask_id, batch_size
                )
    
    def _generate_single_batch(self, model, prompt: torch.Tensor, steps: int,
                              gen_length: int, block_length: int, temperature: float,
                              cfg_scale: float, remasking: str, mask_id: int) -> torch.Tensor:
        """Generate for a single batch with optimized memory usage."""
        
        bs = prompt.shape[0]
        device = prompt.device
        dtype = model.dtype if hasattr(model, 'dtype') else torch.float32
        
        # Initialize generation tensor
        x = torch.full((bs, prompt.shape[1] + gen_length), mask_id, 
                      dtype=torch.long, device=device)
        x[:, :prompt.shape[1]] = prompt.clone()
        
        prompt_index = x != mask_id
        
        # Validate generation parameters
        if gen_length % block_length != 0:
            raise ValueError(f"gen_length ({gen_length}) must be divisible by block_length ({block_length})")
        
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)
        
        with self.memory_manager.managed_forward("generation"):
            for num_block in range(num_blocks):
                start_idx = prompt.shape[1] + num_block * block_length
                end_idx = prompt.shape[1] + (num_block + 1) * block_length
                
                # Pre-compute transfer tokens schedule for this block
                transfer_tokens_schedule = self._precompute_transfer_schedule(
                    bs, block_length, steps_per_block, device
                )
                
                x = self._generate_block(
                    model, x, start_idx, end_idx, steps_per_block,
                    temperature, cfg_scale, remasking, mask_id,
                    prompt_index, transfer_tokens_schedule, dtype
                )
        
        return x
    
    def _generate_multiple_batches(self, model, prompt: torch.Tensor, steps: int,
                                  gen_length: int, block_length: int, temperature: float,
                                  cfg_scale: float, remasking: str, mask_id: int,
                                  batch_size: int) -> torch.Tensor:
        """Generate for multiple batches with optimized memory usage."""
        
        total_samples = prompt.shape[0]
        results = []
        
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            batch_prompt = prompt[i:end_idx]
            
            batch_result = self._generate_single_batch(
                model, batch_prompt, steps, gen_length, block_length,
                temperature, cfg_scale, remasking, mask_id
            )
            
            results.append(batch_result)
            
            # Clean up intermediate results without aggressive cache clearing
            if i > 0:  # Keep first batch for shape reference
                batch_tensor_cleanup(batch_prompt, force_cache_clear=False)
        
        return torch.cat(results, dim=0)
    
    def _generate_block(self, model, x: torch.Tensor, start_idx: int, end_idx: int,
                       steps_per_block: int, temperature: float, cfg_scale: float,
                       remasking: str, mask_id: int, prompt_index: torch.Tensor,
                       transfer_tokens_schedule: List[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        """Generate a single block with optimized operations."""
        
        block_mask_index = x[:, start_idx:end_idx] == mask_id
        
        for step in range(steps_per_block):
            mask_index = x == mask_id
            
            # Get logits with mixed precision if enabled
            if self.enable_mixed_precision:
                with torch.cuda.amp.autocast(enabled=True):
                    logits = self._get_logits_with_cfg(model, x, prompt_index, cfg_scale, mask_id)
            else:
                logits = self._get_logits_with_cfg(model, x, prompt_index, cfg_scale, mask_id)
            
            # Apply temperature and get predictions
            x0 = self._apply_temperature_and_sample(logits, temperature, dtype)
            
            # Handle remasking strategy
            confidence = self._compute_confidence(logits, x0, remasking, mask_index, end_idx, dtype)
            
            # Update tokens based on confidence
            x = self._update_tokens(
                x, x0, confidence, mask_index, transfer_tokens_schedule[step], step
            )
            
            # Clean up intermediate tensors (no cache clearing for performance)
            batch_tensor_cleanup(logits, x0, confidence, force_cache_clear=False)
        
        return x
    
    def _get_logits_with_cfg(self, model, x: torch.Tensor, prompt_index: torch.Tensor,
                            cfg_scale: float, mask_id: int) -> torch.Tensor:
        """Get logits with classifier-free guidance if enabled."""
        
        if cfg_scale > 0.0:
            # Create unconditional input
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            combined_input = torch.cat([x, un_x], dim=0)
            
            # Single forward pass for both conditional and unconditional
            combined_logits = model(combined_input).logits
            logits, un_logits = torch.chunk(combined_logits, 2, dim=0)
            
            # Apply CFG
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            
            # Clean up intermediate tensors (no cache clearing for performance)
            batch_tensor_cleanup(un_x, combined_input, combined_logits, un_logits, force_cache_clear=False)
        else:
            logits = model(x).logits
        
        return logits
    
    def _apply_temperature_and_sample(self, logits: torch.Tensor, temperature: float,
                                     dtype: torch.dtype) -> torch.Tensor:
        """Apply temperature and sample from logits."""
        
        if temperature == 0.0:
            return torch.argmax(logits, dim=-1)
        
        # Apply Gumbel noise for sampling
        logits_with_noise = self._add_gumbel_noise(logits, temperature, dtype)
        x0 = torch.argmax(logits_with_noise, dim=-1)
        
        batch_tensor_cleanup(logits_with_noise, force_cache_clear=False)
        return x0
    
    def _add_gumbel_noise(self, logits: torch.Tensor, temperature: float,
                         dtype: torch.dtype) -> torch.Tensor:
        """Add Gumbel noise for sampling."""
        
        logits = logits.to(dtype)
        noise = torch.rand_like(logits, dtype=dtype)
        gumbel_noise = (-torch.log(noise)) ** temperature
        
        return logits.exp() / gumbel_noise
    
    def _compute_confidence(self, logits: torch.Tensor, x0: torch.Tensor, remasking: str,
                           mask_index: torch.Tensor, end_idx: int, dtype: torch.dtype) -> torch.Tensor:
        """Compute confidence scores for remasking."""
        
        if remasking == "low_confidence":
            p = F.softmax(logits.to(dtype), dim=-1)
            confidence = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
            )
            batch_tensor_cleanup(p, force_cache_clear=False)
        elif remasking == "random":
            confidence = torch.rand_like(x0, dtype=dtype)
        else:
            raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented")
        
        # Mask tokens beyond current block
        confidence[:, end_idx:] = -np.inf
        
        return confidence
    
    def _update_tokens(self, x: torch.Tensor, x0: torch.Tensor, confidence: torch.Tensor,
                      mask_index: torch.Tensor, num_transfer_tokens: torch.Tensor,
                      step: int) -> torch.Tensor:
        """Update tokens based on confidence scores."""
        
        # Only update masked tokens
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, confidence, -np.inf)
        
        # Select tokens to transfer based on confidence
        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        
        # Handle both scalar and tensor cases for num_transfer_tokens
        if num_transfer_tokens.dim() == 0:
            # 0-dimensional tensor (scalar) - same number of tokens for all batch elements
            num_tokens = num_transfer_tokens.item()
            for j in range(confidence.shape[0]):
                if num_tokens > 0:
                    _, select_index = torch.topk(confidence[j], k=num_tokens)
                    transfer_index[j, select_index] = True
        else:
            # 1-dimensional tensor - different number of tokens per batch element
            for j in range(confidence.shape[0]):
                num_tokens = num_transfer_tokens[j].item()
                if num_tokens > 0:
                    _, select_index = torch.topk(confidence[j], k=num_tokens)
                    transfer_index[j, select_index] = True
        
        # Update selected tokens
        x[transfer_index] = x0[transfer_index]
        
        return x
    
    def _precompute_transfer_schedule(self, batch_size: int, block_length: int,
                                     steps_per_block: int, device: torch.device) -> List[torch.Tensor]:
        """Precompute transfer token schedule for all blocks."""
        
        # Assume uniform distribution for now - can be made more sophisticated
        mask_num = torch.full((batch_size, 1), block_length, device=device)
        
        schedule = []
        for step in range(steps_per_block):
            remaining_steps = steps_per_block - step
            base_tokens = mask_num // remaining_steps
            remainder = mask_num % remaining_steps
            
            num_transfer = base_tokens.squeeze(-1)
            if step < remainder.max().item():
                num_transfer += (step < remainder.squeeze(-1)).long()
            
            # Ensure we always have at least 1D tensor even for batch_size=1
            if num_transfer.dim() == 0:
                num_transfer = num_transfer.unsqueeze(0)
            
            schedule.append(num_transfer)
        
        return schedule
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation performance statistics."""
        stats = {
            'memory_stats': self.memory_manager.get_memory_stats(),
            'cache_enabled': self.cache_logits,
            'mixed_precision': self.enable_mixed_precision
        }
        
        if self.cache_logits:
            stats['cache_size'] = len(self._logits_cache)
        
        return stats
    
    def clear_cache(self):
        """Clear all caches."""
        self._logits_cache.clear()
        self.memory_manager.clear_stats()