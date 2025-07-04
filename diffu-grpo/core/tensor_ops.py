import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class TensorOpsOptimizer:
    """Optimized tensor operations with caching and batching."""
    
    def __init__(self, cache_size: int = 1000, device: Optional[torch.device] = None):
        self.cache_size = cache_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._shape_cache = {}
        self._logits_cache = {}
        
    def efficient_cat(self, tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
        """Efficiently concatenate tensors with memory optimization."""
        if not tensors:
            raise ValueError("Cannot concatenate empty list of tensors")
        
        if len(tensors) == 1:
            return tensors[0]
        
        # Pre-allocate result tensor if all shapes are known
        total_size = sum(t.shape[dim] for t in tensors)
        first_tensor = tensors[0]
        result_shape = list(first_tensor.shape)
        result_shape[dim] = total_size
        
        result = torch.empty(result_shape, dtype=first_tensor.dtype, device=first_tensor.device)
        
        # Copy tensors in-place
        current_idx = 0
        for tensor in tensors:
            size = tensor.shape[dim]
            if dim == 0:
                result[current_idx:current_idx + size] = tensor
            elif dim == 1:
                result[:, current_idx:current_idx + size] = tensor
            else:
                # For higher dimensions, fall back to torch.cat
                return torch.cat(tensors, dim=dim)
            current_idx += size
        
        return result
    
    def batched_cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor,
                             reduction: str = 'none', batch_size: Optional[int] = None) -> torch.Tensor:
        """Compute cross entropy in batches to reduce memory usage."""
        if batch_size is None or logits.size(0) <= batch_size:
            return F.cross_entropy(logits, targets, reduction=reduction)
        
        total_samples = logits.size(0)
        results = []
        
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            batch_logits = logits[i:end_idx]
            batch_targets = targets[i:end_idx]
            batch_loss = F.cross_entropy(batch_logits, batch_targets, reduction=reduction)
            results.append(batch_loss)
        
        return torch.cat(results, dim=0)
    
    def efficient_topk(self, tensor: torch.Tensor, k: int, dim: int = -1,
                      sorted: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Memory-efficient topk operation."""
        if k >= tensor.size(dim):
            # If k is large relative to tensor size, use sort instead
            sorted_tensor, indices = torch.sort(tensor, dim=dim, descending=True)
            return sorted_tensor[..., :k], indices[..., :k]
        else:
            return torch.topk(tensor, k, dim=dim, sorted=sorted)
    
    def batched_softmax(self, logits: torch.Tensor, dim: int = -1,
                       batch_size: Optional[int] = None) -> torch.Tensor:
        """Compute softmax in batches for memory efficiency."""
        if batch_size is None or logits.numel() < batch_size * logits.size(-1):
            return F.softmax(logits, dim=dim)
        
        # Reshape for batched processing
        original_shape = logits.shape
        if dim != -1 and dim != len(original_shape) - 1:
            # Move target dim to last position
            dims = list(range(len(original_shape)))
            dims[dim], dims[-1] = dims[-1], dims[dim]
            logits = logits.permute(dims)
            dim = -1
        
        # Flatten all but last dimension
        flat_logits = logits.view(-1, logits.size(-1))
        total_samples = flat_logits.size(0)
        
        results = []
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            batch_logits = flat_logits[i:end_idx]
            batch_softmax = F.softmax(batch_logits, dim=-1)
            results.append(batch_softmax)
        
        result = torch.cat(results, dim=0)
        
        # Reshape back to original shape
        result = result.view(logits.shape)
        
        # Permute back if necessary
        if dim != -1 and dim != len(original_shape) - 1:
            dims = list(range(len(original_shape)))
            dims[dim], dims[-1] = dims[-1], dims[dim]
            result = result.permute(dims)
        
        return result
    
    def efficient_gather(self, input_tensor: torch.Tensor, indices: torch.Tensor,
                        dim: int = -1) -> torch.Tensor:
        """Memory-efficient gather operation."""
        # Check if we can use a more efficient indexing method
        if dim == -1 or dim == len(input_tensor.shape) - 1:
            if indices.dim() == 1:
                # Simple 1D indexing
                return input_tensor[..., indices]
        
        return torch.gather(input_tensor, dim, indices)
    
    def precompute_transfer_tokens(self, mask_index: torch.Tensor, steps: int) -> torch.Tensor:
        """Optimized computation of transfer token schedule."""
        mask_num = mask_index.sum(dim=1, keepdim=True)
        
        # Use more efficient computation
        base = mask_num // steps
        remainder = mask_num % steps
        
        # Create schedule tensor efficiently
        batch_size = mask_index.size(0)
        schedule = base.expand(batch_size, steps).clone()
        
        # Handle remainder efficiently using broadcasting
        step_indices = torch.arange(steps, device=mask_index.device).unsqueeze(0)
        remainder_mask = step_indices < remainder
        schedule[remainder_mask] += 1
        
        return schedule.to(torch.int64)
    
    def efficient_where(self, condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Memory-efficient where operation with broadcasting optimization."""
        # Check if we can use more efficient operations
        if condition.dtype == torch.bool:
            # Use masked_fill when possible
            if torch.is_tensor(y) and y.numel() == 1:
                result = x.clone()
                result.masked_fill_(~condition, y.item())
                return result
            elif torch.is_tensor(x) and x.numel() == 1:
                result = y.clone()
                result.masked_fill_(condition, x.item())
                return result
        
        return torch.where(condition, x, y)
    
    def batched_log_prob_computation(self, logits: torch.Tensor, targets: torch.Tensor,
                                   batch_size: int = 1024) -> torch.Tensor:
        """Compute log probabilities in batches for memory efficiency."""
        total_samples = logits.size(0)
        seq_len = logits.size(1)
        vocab_size = logits.size(2)
        
        if total_samples * seq_len <= batch_size:
            # Small enough to compute all at once
            flat_logits = logits.view(-1, vocab_size)
            flat_targets = targets.view(-1)
            log_probs = F.log_softmax(flat_logits, dim=-1)
            gathered = log_probs.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
            return gathered.view(total_samples, seq_len)
        
        # Process in batches
        results = []
        flat_logits = logits.view(-1, vocab_size)
        flat_targets = targets.view(-1)
        
        for i in range(0, flat_logits.size(0), batch_size):
            end_idx = min(i + batch_size, flat_logits.size(0))
            batch_logits = flat_logits[i:end_idx]
            batch_targets = flat_targets[i:end_idx]
            
            batch_log_probs = F.log_softmax(batch_logits, dim=-1)
            batch_gathered = batch_log_probs.gather(1, batch_targets.unsqueeze(1)).squeeze(1)
            results.append(batch_gathered)
        
        result = torch.cat(results, dim=0)
        return result.view(total_samples, seq_len)
    
    @lru_cache(maxsize=100)
    def get_prompt_index_cached(self, seq_len: int, prompt_len: int, device_str: str) -> torch.Tensor:
        """Cached computation of prompt index tensor."""
        device = torch.device(device_str)
        prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=device)
        prompt_index[:prompt_len] = True
        return prompt_index
    
    def create_attention_mask(self, input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        """Efficiently create attention mask."""
        return (input_ids != pad_token_id).long()
    
    def efficient_masked_fill(self, tensor: torch.Tensor, mask: torch.Tensor, value: float) -> torch.Tensor:
        """Memory-efficient masked fill operation."""
        if mask.sum() == 0:
            return tensor
        
        # Use in-place operation when possible
        if tensor.is_contiguous() and mask.is_contiguous():
            result = tensor.clone()
            result.masked_fill_(mask, value)
            return result
        else:
            return tensor.masked_fill(mask, value)
    
    def clear_cache(self):
        """Clear all caches."""
        self._shape_cache.clear()
        self._logits_cache.clear()
        # Clear LRU cache
        self.get_prompt_index_cached.cache_clear()


class MemoryEfficientOperations:
    """Collection of memory-efficient tensor operations."""
    
    @staticmethod
    def chunk_tensor_operation(tensor: torch.Tensor, operation_fn, chunk_size: int = 1024) -> torch.Tensor:
        """Apply operation to tensor in chunks to reduce memory usage."""
        if tensor.size(0) <= chunk_size:
            return operation_fn(tensor)
        
        results = []
        for i in range(0, tensor.size(0), chunk_size):
            end_idx = min(i + chunk_size, tensor.size(0))
            chunk = tensor[i:end_idx]
            result_chunk = operation_fn(chunk)
            results.append(result_chunk)
        
        return torch.cat(results, dim=0)
    
    @staticmethod
    def gradient_checkpointing_wrapper(model, *args, **kwargs):
        """Wrapper for gradient checkpointing to reduce memory usage."""
        if hasattr(torch.utils.checkpoint, 'checkpoint'):
            return torch.utils.checkpoint.checkpoint(model, *args, **kwargs)
        else:
            return model(*args, **kwargs)
    
    @staticmethod
    def efficient_tensor_split(tensor: torch.Tensor, split_size: int, dim: int = 0) -> List[torch.Tensor]:
        """Split tensor efficiently without creating unnecessary copies."""
        if tensor.size(dim) <= split_size:
            return [tensor]
        
        splits = []
        for i in range(0, tensor.size(dim), split_size):
            end_idx = min(i + split_size, tensor.size(dim))
            if dim == 0:
                split_tensor = tensor[i:end_idx]
            elif dim == 1:
                split_tensor = tensor[:, i:end_idx]
            else:
                # Use torch.split for higher dimensions
                return list(torch.split(tensor, split_size, dim=dim))
            splits.append(split_tensor)
        
        return splits
    
    @staticmethod
    def batch_tensor_copy(src_tensors: List[torch.Tensor], dst_tensors: List[torch.Tensor]):
        """Efficiently copy multiple tensors."""
        if len(src_tensors) != len(dst_tensors):
            raise ValueError("Source and destination tensor lists must have same length")
        
        for src, dst in zip(src_tensors, dst_tensors):
            if src.shape != dst.shape:
                raise ValueError(f"Tensor shape mismatch: {src.shape} vs {dst.shape}")
            dst.copy_(src)


# Global instance for easy access
tensor_ops = TensorOpsOptimizer()