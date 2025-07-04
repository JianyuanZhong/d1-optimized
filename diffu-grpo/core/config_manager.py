import os
import json
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for optimization settings."""
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = False
    enable_profiling: bool = False
    cache_size: int = 1000
    
    # Batch size configurations (all per-device for distributed training)
    generation_batch_size: int = 4  # Per-device batch size for generation phase
    tensor_batch_size: int = 1024   # Per-device batch size for tensor operations
    per_device_train_batch_size: int = 1  # Per-device training batch size
    
    memory_efficient_attention: bool = True


@dataclass
class MaskingConfig:
    """Configuration for masking strategy."""
    strategy_type: str = "diffusion"  # diffusion, optimized, adaptive
    p_mask_prompt: float = 0.3
    cache_size: int = 1000
    num_precomputed: int = 100
    
    # Adaptive masking settings
    initial_p_mask: Optional[float] = None
    final_p_mask: Optional[float] = None
    total_steps: Optional[int] = None


@dataclass
class GenerationConfig:
    """Configuration for diffusion generation."""
    steps: int = 64
    gen_length: int = 128
    block_length: int = 64
    temperature: float = 0.0
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"
    mask_id: int = 126336
    enable_caching: bool = True


@dataclass
class LossConfig:
    """Configuration for GRPO loss computation."""
    epsilon: float = 0.2
    beta: float = 0.04
    adaptive_loss: bool = False
    
    # Adaptive loss settings
    initial_epsilon: Optional[float] = None
    final_epsilon: Optional[float] = None
    initial_beta: Optional[float] = None
    final_beta: Optional[float] = None
    total_steps: Optional[int] = None
    
    # Multi-objective settings
    length_penalty_weight: float = 0.0


@dataclass
class RewardConfig:
    """Configuration for reward functions."""
    use_new_rewards: bool = True
    reward_weights: Optional[List[float]] = None
    enable_logging: bool = False
    log_probability: float = 0.1
    
    # Correctness reward settings
    correct_reward: float = 2.0
    format_reward: float = 0.5


@dataclass
class ImprovedDiffuGRPOConfig:
    """Comprehensive configuration for improved GRPO training."""
    
    # Core training settings (from original config)
    model_path: str = ""
    dataset: str = "gsm8k"
    num_iterations: int = 1
    max_completion_length: int = 256
    max_prompt_length: int = 256
    num_generations: int = 8
    random_masking: bool = True
    
    # Component configurations
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    
    # Logging and monitoring
    enable_performance_logging: bool = True
    log_memory_usage: bool = True
    save_performance_stats: bool = True
    performance_log_interval: int = 100

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self):
        """Validate configuration settings."""
        errors = []
        
        # Validate basic settings
        if self.num_generations <= 0:
            errors.append("num_generations must be positive")
        
        if self.max_completion_length <= 0:
            errors.append("max_completion_length must be positive")
        
        if self.generation.gen_length % self.generation.block_length != 0:
            errors.append("gen_length must be divisible by block_length")
        
        # Validate masking config
        if not 0 <= self.masking.p_mask_prompt <= 1:
            errors.append("p_mask_prompt must be between 0 and 1")
        
        if self.masking.strategy_type == "adaptive":
            if self.masking.initial_p_mask is None or self.masking.final_p_mask is None:
                errors.append("Adaptive masking requires initial_p_mask and final_p_mask")
        
        # Validate loss config
        if self.loss.epsilon <= 0:
            errors.append("epsilon must be positive")
        
        if self.loss.beta < 0:
            errors.append("beta must be non-negative")
        
        if self.loss.adaptive_loss:
            if any(x is None for x in [self.loss.initial_epsilon, self.loss.final_epsilon, 
                                     self.loss.initial_beta, self.loss.final_beta]):
                errors.append("Adaptive loss requires all initial/final values")
        
        # Validate generation config
        if self.generation.steps <= 0:
            errors.append("generation steps must be positive")
        
        if not 0 <= self.generation.temperature <= 2:
            logger.warning("temperature outside typical range [0, 2]")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {', '.join(errors)}")
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), (OptimizationConfig, MaskingConfig, 
                                                 GenerationConfig, LossConfig, RewardConfig)):
                    # Update nested config
                    nested_config = getattr(self, key)
                    if isinstance(value, dict):
                        for nested_key, nested_value in value.items():
                            if hasattr(nested_config, nested_key):
                                setattr(nested_config, nested_key, nested_value)
                else:
                    setattr(self, key, value)
        
        # Re-validate after updates
        self.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save(self, filepath: Union[str, Path]):
        """Save configuration to file."""
        filepath = Path(filepath)
        config_dict = self.to_dict()
        
        if filepath.suffix.lower() == '.json':
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif filepath.suffix.lower() in ['.yaml', '.yml']:
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ImprovedDiffuGRPOConfig':
        """Load configuration from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        if filepath.suffix.lower() == '.json':
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        elif filepath.suffix.lower() in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        # Create config with nested structure
        config = cls()
        config.update_from_dict(config_dict)
        
        logger.info(f"Configuration loaded from {filepath}")
        return config


class ConfigManager:
    """Manager for configuration files and environment overrides."""
    
    def __init__(self, base_config: Optional[ImprovedDiffuGRPOConfig] = None):
        self.base_config = base_config or ImprovedDiffuGRPOConfig()
        self.config_history = []
    
    def load_config(self, config_path: Union[str, Path]) -> ImprovedDiffuGRPOConfig:
        """Load configuration from file with environment overrides."""
        config = ImprovedDiffuGRPOConfig.load(config_path)
        
        # Apply environment variable overrides
        config = self._apply_env_overrides(config)
        
        # Store in history
        self.config_history.append(config.to_dict())
        
        return config
    
    def _apply_env_overrides(self, config: ImprovedDiffuGRPOConfig) -> ImprovedDiffuGRPOConfig:
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            'DIFFU_MODEL_PATH': 'model_path',
            'DIFFU_DATASET': 'dataset',
            'DIFFU_NUM_ITERATIONS': 'num_iterations',
            'DIFFU_BATCH_SIZE': 'optimization.generation_batch_size',  # Per-device generation batch size
            'DIFFU_TRAIN_BATCH_SIZE': 'optimization.per_device_train_batch_size',  # Per-device training batch size
            'DIFFU_TENSOR_BATCH_SIZE': 'optimization.tensor_batch_size',  # Per-device tensor operations batch size
            'DIFFU_MIXED_PRECISION': 'optimization.enable_mixed_precision',
            'DIFFU_P_MASK_PROMPT': 'masking.p_mask_prompt',
            'DIFFU_GENERATION_STEPS': 'generation.steps',
            'DIFFU_EPSILON': 'loss.epsilon',
            'DIFFU_BETA': 'loss.beta',
            'DIFFU_ENABLE_PROFILING': 'optimization.enable_profiling',
        }
        
        config_dict = config.to_dict()
        
        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Convert value to appropriate type
                if config_path.endswith(('enable_mixed_precision', 'enable_profiling', 'random_masking')):
                    value = value.lower() in ('true', '1', 'yes')
                elif config_path.endswith(('num_iterations', 'generation_batch_size', 'per_device_train_batch_size', 'tensor_batch_size', 'steps')):
                    value = int(value)
                elif config_path.endswith(('p_mask_prompt', 'epsilon', 'beta')):
                    value = float(value)
                
                # Set nested value
                self._set_nested_value(config_dict, config_path, value)
                logger.info(f"Applied environment override: {env_var}={value}")
        
        # Create new config from updated dict
        new_config = ImprovedDiffuGRPOConfig()
        new_config.update_from_dict(config_dict)
        
        return new_config
    
    def _set_nested_value(self, config_dict: Dict[str, Any], path: str, value: Any):
        """Set nested value in configuration dictionary."""
        keys = path.split('.')
        current = config_dict
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def create_dataset_config(self, dataset: str, base_config: Optional[ImprovedDiffuGRPOConfig] = None) -> ImprovedDiffuGRPOConfig:
        """Create optimized configuration for specific dataset."""
        config = base_config or ImprovedDiffuGRPOConfig()
        config.dataset = dataset
        
        # Dataset-specific optimizations
        if dataset == "gsm8k":
            config.generation.gen_length = 256
            config.generation.block_length = 64
            config.masking.p_mask_prompt = 0.3
            config.loss.epsilon = 0.2
            config.num_generations = 8
        
        elif dataset == "countdown":
            config.generation.gen_length = 128
            config.generation.block_length = 32
            config.masking.p_mask_prompt = 0.2
            config.loss.epsilon = 0.15
            config.num_generations = 4
        
        elif dataset == "sudoku":
            config.generation.gen_length = 64
            config.generation.block_length = 16
            config.masking.p_mask_prompt = 0.4
            config.loss.epsilon = 0.25
            config.num_generations = 6
        
        elif dataset == "math":
            config.generation.gen_length = 512
            config.generation.block_length = 128
            config.masking.p_mask_prompt = 0.3
            config.loss.epsilon = 0.2
            config.num_generations = 8
        
        config.validate()
        return config
    
    def get_performance_config(self, optimization_level: str = "balanced") -> ImprovedDiffuGRPOConfig:
        """Get configuration optimized for performance."""
        config = ImprovedDiffuGRPOConfig()
        
        if optimization_level == "speed":
            # Optimize for training speed
            config.optimization.enable_mixed_precision = True
            config.optimization.generation_batch_size = 8
            config.optimization.cache_size = 2000
            config.masking.strategy_type = "optimized"
            config.generation.enable_caching = True
            config.loss.adaptive_loss = False
            
        elif optimization_level == "memory":
            # Optimize for memory usage
            config.optimization.enable_gradient_checkpointing = True
            config.optimization.generation_batch_size = 2
            config.optimization.cache_size = 500
            config.optimization.tensor_batch_size = 512
            config.masking.cache_size = 500
            
        elif optimization_level == "balanced":
            # Balanced optimization
            config.optimization.enable_mixed_precision = True
            config.optimization.generation_batch_size = 4
            config.optimization.cache_size = 1000
            config.masking.strategy_type = "diffusion"
            config.generation.enable_caching = True
        
        return config
    
    def save_experiment_config(self, config: ImprovedDiffuGRPOConfig, 
                             experiment_name: str, output_dir: str):
        """Save configuration for experiment reproducibility."""
        output_path = Path(output_dir) / f"{experiment_name}_config.yaml"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        config_dict = config.to_dict()
        config_dict['_metadata'] = {
            'experiment_name': experiment_name,
            'created_at': str(Path(__file__).stat().st_mtime),
            'config_version': '1.0'
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Experiment configuration saved to {output_path}")


# Factory functions for common configurations
def create_gsm8k_config() -> ImprovedDiffuGRPOConfig:
    """Create optimized configuration for GSM8K dataset."""
    manager = ConfigManager()
    return manager.create_dataset_config("gsm8k")


def create_countdown_config() -> ImprovedDiffuGRPOConfig:
    """Create optimized configuration for Countdown dataset."""
    manager = ConfigManager()
    return manager.create_dataset_config("countdown")


def create_speed_optimized_config() -> ImprovedDiffuGRPOConfig:
    """Create speed-optimized configuration."""
    manager = ConfigManager()
    return manager.get_performance_config("speed")


def create_memory_optimized_config() -> ImprovedDiffuGRPOConfig:
    """Create memory-optimized configuration."""
    manager = ConfigManager()
    return manager.get_performance_config("memory")