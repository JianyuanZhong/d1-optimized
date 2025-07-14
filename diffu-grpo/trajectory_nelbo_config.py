"""
Configuration example for using TrajectoryNELBOLoss with ImprovedDiffuGRPOTrainer.

This file demonstrates how to configure the trajectory-aware NELBO method for 
estimating log probabilities in discrete diffusion models.
"""

from diffu_grpo_config import DiffuGRPOConfig


class TrajectoryNELBOConfig(DiffuGRPOConfig):
    """Configuration for trajectory-aware NELBO training."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Enable trajectory NELBO loss
        self.trajectory_nelbo_loss = True
        
        # Discrete diffusion parameters aligned with NELBO theory
        self.vocab_size = 32000  # Model vocabulary size
        self.diffusion_steps = 128  # Number of timesteps T in the NELBO formulation
        self.alpha_schedule = "cosine"  # Noise schedule: "cosine", "linear", "exponential"
        self.trajectory_samples = 4  # Number of trajectory samples for NELBO estimation
        
        # NELBO-specific parameters
        self.alpha_min = 0.01  # Minimum α value (maximum noise)
        self.alpha_max = 0.99  # Maximum α value (minimum noise)
        self.prior_type = "uniform"  # Prior distribution π: "uniform" or "mask_focused"
        
        # Masking consistency
        self.mask_id = 126336  # Must match model's mask token ID
        self.p_mask_prompt = 0.3  # Prompt masking probability (consistent with DiffusionMaskingStrategy)
        
        # Generation parameters (should align with NELBO timesteps)
        self.block_length = 64  # Generation block size
        self.temperature = 0.0  # Generation temperature
        self.cfg_scale = 0.0  # Classifier-free guidance scale
        self.remasking = "low_confidence"  # Remasking strategy
        
        # Training parameters
        self.beta = 0.04  # KL regularization coefficient
        self.epsilon = 0.2  # PPO clipping parameter
        self.num_iterations = 4  # Number of GRPO iterations per batch
        
        # Memory optimization
        self.fp16 = True  # Use mixed precision
        self.memory_cache_interval = 10  # Cache clearing frequency
        
        # Logging and evaluation
        self.log_completions = True
        self.logging_steps = 100
        
    def validate_config(self):
        """Validate trajectory NELBO configuration."""
        assert self.trajectory_nelbo_loss, "trajectory_nelbo_loss must be True"
        assert self.diffusion_steps > 0, "diffusion_steps must be positive"
        assert 0 < self.alpha_min < self.alpha_max < 1, "Invalid alpha range"
        assert self.trajectory_samples > 0, "trajectory_samples must be positive"
        assert self.alpha_schedule in ["cosine", "linear", "exponential"], f"Unknown alpha_schedule: {self.alpha_schedule}"
        assert self.prior_type in ["uniform", "mask_focused"], f"Unknown prior_type: {self.prior_type}"
        
        # Ensure consistency between generation and NELBO parameters
        if hasattr(self, 'max_completion_length') and hasattr(self, 'block_length'):
            assert self.max_completion_length % self.block_length == 0, \
                "max_completion_length must be divisible by block_length"
        
        print("✓ TrajectoryNELBOConfig validation passed")


def create_trajectory_nelbo_config(**kwargs):
    """Factory function to create a validated trajectory NELBO configuration."""
    config = TrajectoryNELBOConfig(**kwargs)
    config.validate_config()
    return config


# Example configurations for different use cases

def get_memory_efficient_config():
    """Memory-efficient configuration for smaller GPUs."""
    return create_trajectory_nelbo_config(
        trajectory_samples=2,  # Reduced for memory
        diffusion_steps=64,    # Reduced timesteps
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        fp16=True,
        memory_cache_interval=5
    )


def get_high_accuracy_config():
    """High-accuracy configuration for larger models/GPUs."""
    return create_trajectory_nelbo_config(
        trajectory_samples=8,   # More samples for better estimation
        diffusion_steps=256,    # More timesteps
        alpha_schedule="cosine", # Smoother transitions
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        bf16=True  # Better precision than fp16
    )


def get_experimental_config():
    """Experimental configuration with mask-focused prior."""
    return create_trajectory_nelbo_config(
        prior_type="mask_focused",  # Prior that emphasizes mask token
        alpha_schedule="exponential",
        trajectory_samples=6,
        alpha_min=0.005,  # Lower minimum for more aggressive noising
        alpha_max=0.995   # Higher maximum for cleaner initial state
    )


if __name__ == "__main__":
    # Test configurations
    print("Testing TrajectoryNELBO configurations...")
    
    # Test memory-efficient config
    config1 = get_memory_efficient_config()
    print(f"Memory-efficient config: {config1.trajectory_samples} samples, {config1.diffusion_steps} steps")
    
    # Test high-accuracy config  
    config2 = get_high_accuracy_config()
    print(f"High-accuracy config: {config2.trajectory_samples} samples, {config2.diffusion_steps} steps")
    
    # Test experimental config
    config3 = get_experimental_config()
    print(f"Experimental config: {config3.prior_type} prior, {config3.alpha_schedule} schedule")
    
    print("All configurations validated successfully!")