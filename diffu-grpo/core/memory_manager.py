import torch
from contextlib import contextmanager
from typing import Optional, List, Any
import gc
import logging
import time

logger = logging.getLogger(__name__)


class MemoryManager:
    """Efficient GPU memory management for diffusion training with smart cache clearing."""
    
    def __init__(self, enable_profiling: bool = False, cache_clear_interval: int = 50):
        self.enable_profiling = enable_profiling
        self._memory_stats = []
        self.cache_clear_interval = cache_clear_interval
        self._operation_count = 0
        self._last_cache_clear = time.time()
        self._memory_threshold = 0.9  # Clear cache when > 90% memory usage
        
    @contextmanager
    def managed_forward(self, operation_name: str = "forward"):
        """Context manager for memory-efficient forward passes with smart cache clearing."""
        if self.enable_profiling:
            initial_memory = torch.cuda.memory_allocated()
            logger.debug(f"[{operation_name}] Initial memory: {initial_memory / 1e9:.2f}GB")
        
        try:
            yield
        finally:
            self._operation_count += 1
            
            # Only clear cache periodically or when memory is high
            should_clear = self._should_clear_cache()
            if should_clear:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._last_cache_clear = time.time()
                if self.enable_profiling:
                    logger.debug(f"[{operation_name}] Cache cleared at operation {self._operation_count}")
            
            if self.enable_profiling:
                final_memory = torch.cuda.memory_allocated()
                logger.debug(f"[{operation_name}] Final memory: {final_memory / 1e9:.2f}GB")
                self._memory_stats.append({
                    'operation': operation_name,
                    'initial': initial_memory,
                    'final': final_memory,
                    'delta': final_memory - initial_memory,
                    'cache_cleared': should_clear
                })
    
    @contextmanager
    def managed_generation(self, batch_size: int, sequence_length: int):
        """Context manager for memory-efficient generation with reduced cache clearing."""
        estimated_memory = batch_size * sequence_length * 4  # rough estimate
        
        if self.enable_profiling:
            logger.debug(f"Starting generation: batch_size={batch_size}, seq_len={sequence_length}")
            logger.debug(f"Estimated memory: {estimated_memory / 1e9:.2f}GB")
        
        # Clear cache before large operations like generation
        initial_clear = self._should_clear_cache_for_large_op(estimated_memory)
        if initial_clear and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self._last_cache_clear = time.time()
        
        try:
            yield
        finally:
            # Only clear after generation if we're using a lot of memory
            if torch.cuda.is_available():
                memory_usage = self._get_memory_usage_ratio()
                if memory_usage > self._memory_threshold:
                    gc.collect()
                    torch.cuda.empty_cache()
                    self._last_cache_clear = time.time()
                    if self.enable_profiling:
                        logger.debug(f"Post-generation cache clear: memory_usage={memory_usage:.2f}")
    
    @contextmanager
    def managed_tensor_ops(self, tensors: List[torch.Tensor], operation_name: str = "tensor_ops"):
        """Context manager for tensor operations with minimal cache clearing."""
        try:
            yield
        finally:
            # Clean up tensors that are no longer needed
            for tensor in tensors:
                if hasattr(tensor, 'data'):
                    del tensor
            
            # Only clear cache if we really need to
            if self._operation_count % (self.cache_clear_interval * 2) == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self._last_cache_clear = time.time()
    
    def _should_clear_cache(self) -> bool:
        """Determine if cache should be cleared based on multiple factors."""
        if not torch.cuda.is_available():
            return False
            
        # Clear based on operation count
        if self._operation_count % self.cache_clear_interval == 0:
            return True
            
        # Clear based on time (avoid clearing too frequently)
        time_since_clear = time.time() - self._last_cache_clear
        if time_since_clear < 5.0:  # Don't clear more than once every 5 seconds
            return False
            
        # Clear based on memory usage
        memory_usage = self._get_memory_usage_ratio()
        if memory_usage > self._memory_threshold:
            return True
            
        return False
    
    def _should_clear_cache_for_large_op(self, estimated_memory: int) -> bool:
        """Determine if cache should be cleared before a large operation."""
        if not torch.cuda.is_available():
            return False
            
        available_memory = torch.cuda.get_device_properties(0).total_memory
        current_memory = torch.cuda.memory_allocated()
        
        # Clear if the operation might cause OOM
        if (current_memory + estimated_memory) > (available_memory * 0.85):
            return True
            
        return False
    
    def _get_memory_usage_ratio(self) -> float:
        """Get current memory usage as a ratio of total available memory."""
        if not torch.cuda.is_available():
            return 0.0
            
        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        return allocated / total
    
    def get_memory_stats(self) -> List[dict]:
        """Get collected memory statistics."""
        return self._memory_stats
    
    def clear_stats(self):
        """Clear collected memory statistics."""
        self._memory_stats.clear()
        self._operation_count = 0
    
    def log_memory_usage(self, operation: str = "current"):
        """Log current memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            usage_ratio = allocated / total
            logger.info(f"[{operation}] GPU Memory - Allocated: {allocated / 1e9:.2f}GB ({usage_ratio:.1%}), Cached: {cached / 1e9:.2f}GB")
    
    def force_cache_clear(self):
        """Force immediate cache clearing (use sparingly)."""
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            self._last_cache_clear = time.time()
            if self.enable_profiling:
                logger.debug("Forced cache clear")


def efficient_tensor_creation(shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Create tensors with memory-efficient allocation."""
    with torch.cuda.device(device):
        return torch.empty(shape, dtype=dtype, device=device, pin_memory=False)


def batch_tensor_cleanup(*tensors: torch.Tensor, force_cache_clear: bool = False):
    """Efficient cleanup of multiple tensors with optional cache clearing."""
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
            del tensor
    
    # Only force cache clear if explicitly requested
    if force_cache_clear and torch.cuda.is_available():
        torch.cuda.empty_cache()