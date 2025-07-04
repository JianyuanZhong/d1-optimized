"""
Core components for improved diffusion GRPO training.

This module provides modular, optimized components for diffusion language model training:
- Memory management utilities
- Diffusion generation with optimized batching
- Flexible masking strategies with caching
- Efficient GRPO loss computation
- Modular reward function framework
- Optimized tensor operations
- Configuration management system
"""

from .memory_manager import MemoryManager, batch_tensor_cleanup
from .diffusion_generator import DiffusionGenerator
from .masking_strategy import (
    MaskingStrategy, DiffusionMaskingStrategy, 
    OptimizedMaskingStrategy, AdaptiveMaskingStrategy,
    create_masking_strategy
)
from .grpo_loss import GRPOLoss, AdaptiveGRPOLoss, MultiObjectiveGRPOLoss
from .reward_functions import (
    RewardFunction, RewardFunctionManager,
    FormatRewardFunction, CorrectnessRewardFunction,
    create_reward_functions
)
from .tensor_ops import TensorOpsOptimizer, MemoryEfficientOperations, tensor_ops
from .config_manager import (
    ImprovedDiffuGRPOConfig, ConfigManager,
    create_gsm8k_config, create_countdown_config,
    create_speed_optimized_config, create_memory_optimized_config
)

__version__ = "1.0.0"
__author__ = "Improved GRPO Team"

__all__ = [
    # Memory management
    "MemoryManager",
    "batch_tensor_cleanup",
    
    # Generation
    "DiffusionGenerator",
    
    # Masking
    "MaskingStrategy",
    "DiffusionMaskingStrategy",
    "OptimizedMaskingStrategy", 
    "AdaptiveMaskingStrategy",
    "create_masking_strategy",
    
    # Loss computation
    "GRPOLoss",
    "AdaptiveGRPOLoss",
    "MultiObjectiveGRPOLoss",
    
    # Reward functions
    "RewardFunction",
    "RewardFunctionManager",
    "FormatRewardFunction",
    "CorrectnessRewardFunction",
    "create_reward_functions",
    
    # Tensor operations
    "TensorOpsOptimizer",
    "MemoryEfficientOperations",
    "tensor_ops",
    
    # Configuration
    "ImprovedDiffuGRPOConfig",
    "ConfigManager",
    "create_gsm8k_config",
    "create_countdown_config", 
    "create_speed_optimized_config",
    "create_memory_optimized_config",
]