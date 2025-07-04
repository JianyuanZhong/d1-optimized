import torch
from contextlib import contextmanager
from typing import Optional, List, Any
import gc
import logging

logger = logging.getLogger(__name__)


class MemoryManager:
    """Efficient GPU memory management for diffusion training."""
    
    def __init__(self, enable_profiling: bool = False):
        self.enable_profiling = enable_profiling
        self._memory_stats = []
        
    @contextmanager
    def managed_forward(self, operation_name: str = "forward"):
        """Context manager for memory-efficient forward passes."""
        if self.enable_profiling:
            initial_memory = torch.cuda.memory_allocated()
            logger.debug(f"[{operation_name}] Initial memory: {initial_memory / 1e9:.2f}GB")
        
        try:
            yield
        finally:
            # Clean up intermediate tensors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if self.enable_profiling:
                final_memory = torch.cuda.memory_allocated()
                logger.debug(f"[{operation_name}] Final memory: {final_memory / 1e9:.2f}GB")
                self._memory_stats.append({
                    'operation': operation_name,
                    'initial': initial_memory,
                    'final': final_memory,
                    'delta': final_memory - initial_memory
                })
    
    @contextmanager
    def managed_generation(self, batch_size: int, sequence_length: int):
        """Context manager for memory-efficient generation."""
        estimated_memory = batch_size * sequence_length * 4  # rough estimate
        
        if self.enable_profiling:
            logger.debug(f"Starting generation: batch_size={batch_size}, seq_len={sequence_length}")
            logger.debug(f"Estimated memory: {estimated_memory / 1e9:.2f}GB")
        
        try:
            yield
        finally:
            # Force garbage collection and clear cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    @contextmanager
    def managed_tensor_ops(self, tensors: List[torch.Tensor], operation_name: str = "tensor_ops"):
        """Context manager for tensor operations with automatic cleanup."""
        try:
            yield
        finally:
            # Clean up tensors that are no longer needed
            for tensor in tensors:
                if hasattr(tensor, 'data'):
                    del tensor
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_memory_stats(self) -> List[dict]:
        """Get collected memory statistics."""
        return self._memory_stats
    
    def clear_stats(self):
        """Clear collected memory statistics."""
        self._memory_stats.clear()
    
    def log_memory_usage(self, operation: str = "current"):
        """Log current memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()
            logger.info(f"[{operation}] GPU Memory - Allocated: {allocated / 1e9:.2f}GB, Cached: {cached / 1e9:.2f}GB")


def efficient_tensor_creation(shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Create tensors with memory-efficient allocation."""
    with torch.cuda.device(device):
        return torch.empty(shape, dtype=dtype, device=device, pin_memory=False)


def batch_tensor_cleanup(*tensors: torch.Tensor):
    """Efficient cleanup of multiple tensors."""
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
            del tensor
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()