"""
Configuration example for using TrajectoryAwareGRPOLoss with ImprovedDiffuGRPOTrainer.

This file demonstrates how to configure the trajectory-aware reinforcement learning approach
that implements importance weighting and outcome supervision for diffusion models.
"""

from diffu_grpo_config import DiffuGRPOConfig


class TrajectoryAwareGRPOConfig(DiffuGRPOConfig):
    """Configuration for trajectory-aware GRPO training."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Enable trajectory-aware GRPO loss
        self.trajectory_aware_grpo_loss = True
        
        # Trajectory-aware specific parameters
        self.importance_weight_normalization = "softmax"  # Normalization strategy for importance weights
        self.per_step_kl_penalty = True  # Apply KL penalty per-step
        self.numerical_stability_eps = 1e-8  # Numerical stability epsilon
        self.max_importance_weight = 10.0  # Maximum importance weight value
        
        # Training parameters optimized for trajectory-aware approach
        self.beta = 0.04  # KL regularization coefficient
        self.epsilon = 0.2  # PPO clipping parameter
        self.num_iterations = 4  # Number of GRPO iterations per batch
        
        # Generation parameters
        self.diffusion_steps = 128  # Number of timesteps for diffusion process
        self.block_length = 64  # Generation block size
        self.temperature = 0.0  # Generation temperature
        self.cfg_scale = 0.0  # Classifier-free guidance scale
        self.remasking = "low_confidence"  # Remasking strategy
        
        # Masking parameters
        self.mask_id = 126336  # Must match model's mask token ID
        self.p_mask_prompt = 0.3  # Prompt masking probability
        
        # Memory optimization
        self.fp16 = True  # Use mixed precision
        self.memory_cache_interval = 10  # Cache clearing frequency
        self.generation_batch_size = 2  # Reduced for memory efficiency
        
        # Logging and evaluation
        self.log_completions = True
        self.logging_steps = 50
        
    def validate_config(self):
        """Validate trajectory-aware GRPO configuration."""
        assert self.trajectory_aware_grpo_loss, "trajectory_aware_grpo_loss must be True"
        assert self.importance_weight_normalization in ["softmax", "clamp", "normalize", "none"], \
            f"Invalid importance_weight_normalization: {self.importance_weight_normalization}"
        assert self.numerical_stability_eps > 0, "numerical_stability_eps must be positive"
        assert self.max_importance_weight > 0, "max_importance_weight must be positive"
        assert self.beta > 0, "beta must be positive for numerical stability"
        assert 0 < self.epsilon <= 1, "epsilon must be in (0, 1]"
        
        # Ensure consistency between generation and training parameters
        if hasattr(self, 'max_completion_length') and hasattr(self, 'block_length'):
            assert self.max_completion_length % self.block_length == 0, \
                "max_completion_length must be divisible by block_length"
        
        print("✓ TrajectoryAwareGRPOConfig validation passed")


def create_trajectory_aware_grpo_config(**kwargs):
    """Factory function to create a validated trajectory-aware GRPO configuration."""
    config = TrajectoryAwareGRPOConfig(**kwargs)
    config.validate_config()
    return config


# Example configurations for different use cases

def get_memory_efficient_config():
    """Memory-efficient configuration for smaller GPUs."""
    return create_trajectory_aware_grpo_config(
        importance_weight_normalization="clamp",  # Less memory than softmax
        max_importance_weight=5.0,  # Lower bound for stability
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        generation_batch_size=1,
        fp16=True
    )


def get_high_accuracy_config():
    """High-accuracy configuration for larger models/GPUs."""
    return create_trajectory_aware_grpo_config(
        importance_weight_normalization="softmax",  # Better normalization
        max_importance_weight=20.0,  # Allow higher importance weights
        numerical_stability_eps=1e-10,  # Higher precision
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        generation_batch_size=4,
        bf16=True  # Better precision than fp16
    )


def get_fast_training_config():
    """Configuration optimized for faster training."""
    return create_trajectory_aware_grpo_config(
        importance_weight_normalization="clamp",  # Fastest normalization
        per_step_kl_penalty=False,  # Disable per-step KL for speed
        num_iterations=2,  # Fewer iterations per batch
        diffusion_steps=64,  # Fewer diffusion steps
        block_length=32  # Smaller blocks
    )


def get_experimental_config():
    """Experimental configuration with different normalization."""
    return create_trajectory_aware_grpo_config(
        importance_weight_normalization="normalize",  # L1 normalization
        per_step_kl_penalty=True,
        max_importance_weight=15.0,
        numerical_stability_eps=1e-6,
        beta=0.06  # Higher KL coefficient
    )


def get_gsm8k_config():
    """Optimized configuration for GSM8K dataset."""
    return create_trajectory_aware_grpo_config(
        dataset="gsm8k",
        max_completion_length=256,
        block_length=64,
        importance_weight_normalization="softmax",
        per_step_kl_penalty=True,
        learning_rate=1e-6,
        eval_steps=250
    )


def get_math_config():
    """Optimized configuration for MATH dataset."""
    return create_trajectory_aware_grpo_config(
        dataset="math",
        max_completion_length=512,
        block_length=128,
        importance_weight_normalization="softmax",
        per_step_kl_penalty=True,
        max_importance_weight=20.0,  # Allow higher weights for complex problems
        learning_rate=5e-7  # Lower learning rate for stability
    )


if __name__ == "__main__":
    # Test configurations
    print("Testing TrajectoryAware GRPO configurations...")
    
    # Test memory-efficient config
    config1 = get_memory_efficient_config()
    print(f"Memory-efficient config: {config1.importance_weight_normalization} normalization, "
          f"max_weight={config1.max_importance_weight}")
    
    # Test high-accuracy config  
    config2 = get_high_accuracy_config()
    print(f"High-accuracy config: {config2.importance_weight_normalization} normalization, "
          f"eps={config2.numerical_stability_eps}")
    
    # Test fast training config
    config3 = get_fast_training_config()
    print(f"Fast training config: {config3.num_iterations} iterations, "
          f"per_step_kl={config3.per_step_kl_penalty}")
    
    # Test dataset-specific configs
    config4 = get_gsm8k_config()
    print(f"GSM8K config: dataset={config4.dataset}, "
          f"completion_length={config4.max_completion_length}")
    
    config5 = get_math_config()
    print(f"MATH config: dataset={config5.dataset}, "
          f"completion_length={config5.max_completion_length}")
    
    print("All configurations validated successfully!")