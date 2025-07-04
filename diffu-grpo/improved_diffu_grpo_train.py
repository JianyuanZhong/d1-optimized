import torch
import wandb
import logging
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from trl import TrlParser, ModelConfig
from peft import LoraConfig

# Import improved components
from improved_diffu_grpo_trainer import ImprovedDiffuGRPOTrainer
from core.config_manager import (
    ImprovedDiffuGRPOConfig, ConfigManager, 
    create_gsm8k_config, create_countdown_config, 
    create_speed_optimized_config, create_memory_optimized_config
)
from core.reward_functions import create_reward_functions, RewardFunctionManager
from data_utils import (
    get_gsm8k_questions,
    get_countdown_questions,
    get_sudoku_questions,
    set_random_seed,
    get_math_questions,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def setup_model_and_tokenizer(config: ImprovedDiffuGRPOConfig, model_config: ModelConfig):
    """Set up model and tokenizer with optimized configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configure quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # Load model with optimizations
    logger.info(f"Loading model from {config.model_path}")
    model = AutoModel.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    ).to(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Optimize model configuration
    model.config.use_cache = False
    if config.optimization.enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")
    
    return model, tokenizer


def setup_peft_config(model_config: ModelConfig):
    """Set up PEFT configuration for efficient fine-tuning."""
    return LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=model_config.lora_dropout,
    )


def load_dataset(config: ImprovedDiffuGRPOConfig):
    """Load and prepare dataset based on configuration."""
    logger.info(f"Loading dataset: {config.dataset}")
    
    if config.dataset == "gsm8k":
        dataset = get_gsm8k_questions("train")
    elif config.dataset == "countdown":
        dataset = get_countdown_questions("train")
    elif config.dataset == "sudoku":
        dataset = get_sudoku_questions()
    elif config.dataset == "math":
        dataset = get_math_questions("train")
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
    
    # Shuffle dataset with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=42)
    
    # Split dataset if needed
    if config.dataset in ["countdown", "sudoku"]:
        train_set = dataset.select(range(0, len(dataset) - 500))
        logger.info(f"Using {len(train_set)} samples for training, reserving 500 for evaluation")
    else:
        train_set = dataset
        logger.info(f"Using {len(train_set)} samples for training")
    
    return train_set


def setup_reward_functions(config: ImprovedDiffuGRPOConfig):
    """Set up reward functions based on configuration."""
    if config.reward.use_new_rewards:
        logger.info("Using new modular reward functions")
        reward_functions = create_reward_functions(config.dataset)
        
        # Apply configuration settings
        for reward_func in reward_functions:
            reward_func.enable_logging = config.reward.enable_logging
            reward_func.log_probability = config.reward.log_probability
            
            # Update reward values if specified
            if hasattr(reward_func, 'correct_reward'):
                reward_func.correct_reward = config.reward.correct_reward
            if hasattr(reward_func, 'reward_value'):
                reward_func.reward_value = config.reward.format_reward
        
        return reward_functions
    else:
        logger.info("Using legacy reward functions")
        # Import legacy reward functions
        from reward_func import (
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
            countdown_reward_func,
            correctness_reward_func_math,
            sudoku_reward_func,
            boxed_and_answer_tags_format_reward,
        )
        
        if config.dataset == "gsm8k":
            return [
                xmlcount_reward_func,
                soft_format_reward_func,
                strict_format_reward_func,
                int_reward_func,
                correctness_reward_func,
            ]
        elif config.dataset == "countdown":
            return [countdown_reward_func]
        elif config.dataset == "sudoku":
            return [sudoku_reward_func]
        elif config.dataset == "math":
            return [
                correctness_reward_func_math,
                boxed_and_answer_tags_format_reward,
            ]


def create_trainer_config(config: ImprovedDiffuGRPOConfig, model_config: ModelConfig, output_dir: str):
    """Create trainer configuration from improved config."""
    # Convert our config to the format expected by the trainer
    from diffu_grpo_config import DiffuGRPOConfig
    
    trainer_config = DiffuGRPOConfig(
        # Basic settings
        model_path=config.model_path,
        dataset=config.dataset,
        num_iterations=config.num_iterations,
        max_completion_length=config.max_completion_length,
        max_prompt_length=config.max_prompt_length,
        num_generations=config.num_generations,
        random_masking=config.random_masking,
        
        # Generation settings
        block_length=config.generation.block_length,
        diffusion_steps=config.generation.steps,
        temperature=config.generation.temperature,
        cfg_scale=config.generation.cfg_scale,
        remasking=config.generation.remasking,
        mask_id=config.generation.mask_id,
        generation_batch_size=config.optimization.generation_batch_size,
        
        # Training settings
        epsilon=config.loss.epsilon,
        beta=config.loss.beta,
        p_mask_prompt=config.masking.p_mask_prompt,
        
        # Optimization settings
        fp16=config.optimization.enable_mixed_precision and not torch.cuda.is_bf16_supported(),
        bf16=config.optimization.enable_mixed_precision and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=config.optimization.enable_gradient_checkpointing,
        
        # Output settings
        output_dir=output_dir,
        logging_steps=config.performance_log_interval,
        save_steps=500,
        eval_steps=500,
        
        # Standard training settings (per-device for distributed training)
        per_device_train_batch_size=config.optimization.per_device_train_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=1e-6,
        max_steps=1000,
        warmup_steps=100,
        remove_unused_columns=False,
        report_to=["wandb"] if wandb.run else [],
    )
    
    return trainer_config


def main():
    """Main training function with improved architecture."""
    logger.info("Starting improved GRPO training")
    
    # Parse command line arguments
    parser = TrlParser((ImprovedDiffuGRPOConfig, ModelConfig))
    try:
        improved_config, model_config = parser.parse_args_and_config()
    except:
        # Fallback to creating default config
        logger.warning("Could not parse config, using defaults")
        improved_config = create_gsm8k_config()  # Default config
        model_config = ModelConfig()
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Load configuration based on dataset and optimization preferences
    config_manager = ConfigManager()
    
    # Create dataset-specific configuration if not provided
    if not hasattr(improved_config, 'model_path') or not improved_config.model_path:
        if hasattr(improved_config, 'dataset'):
            dataset = improved_config.dataset
        else:
            dataset = "gsm8k"  # Default
            
        logger.info(f"Creating optimized config for dataset: {dataset}")
        improved_config = config_manager.create_dataset_config(dataset)
    
    # Apply performance optimizations based on available resources
    available_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    if available_memory < 16 * 1024**3:  # Less than 16GB
        logger.info("Applying memory optimizations for limited VRAM")
        memory_config = create_memory_optimized_config()
        improved_config.optimization = memory_config.optimization
    
    # Set model path if provided via environment or command line
    import os
    if 'DIFFU_MODEL_PATH' in os.environ:
        improved_config.model_path = os.environ['DIFFU_MODEL_PATH']
    elif not improved_config.model_path:
        improved_config.model_path = "/data0/shared/LLaDA-8B-Instruct"  # Default
    
    logger.info("Configuration:")
    logger.info(f"  Model: {improved_config.model_path}")
    logger.info(f"  Dataset: {improved_config.dataset}")
    logger.info(f"  Num iterations: {improved_config.num_iterations}")
    logger.info(f"  Mixed precision: {improved_config.optimization.enable_mixed_precision}")
    logger.info(f"  Generation batch size: {improved_config.optimization.generation_batch_size}")
    
    # Save configuration for reproducibility
    output_dir = f"checkpoints/{improved_config.dataset}_improved_bs{improved_config.optimization.generation_batch_size}"
    config_manager.save_experiment_config(improved_config, f"{improved_config.dataset}_improved", output_dir)
    
    # Set up model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(improved_config, model_config)
    
    # Load dataset
    train_dataset = load_dataset(improved_config)
    
    # Set up reward functions
    reward_functions = setup_reward_functions(improved_config)
    
    # Set up PEFT configuration
    peft_config = setup_peft_config(model_config)
    
    # Create trainer configuration
    trainer_config = create_trainer_config(improved_config, model_config, output_dir)
    
    # Initialize improved trainer
    logger.info("Initializing improved GRPO trainer")
    trainer = ImprovedDiffuGRPOTrainer(
        args=trainer_config,
        model=model,
        peft_config=peft_config,
        reward_funcs=reward_functions,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        enable_profiling=improved_config.optimization.enable_profiling,
        masking_strategy=improved_config.masking.strategy_type,
        adaptive_loss=improved_config.loss.adaptive_loss,
    )
    
    # Set up performance monitoring
    if improved_config.enable_performance_logging:
        def log_performance_callback():
            if trainer.state.global_step % improved_config.performance_log_interval == 0:
                trainer.log_performance_stats()
        
        # Add custom callback for performance logging
        from transformers import TrainerCallback
        
        class PerformanceCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step % improved_config.performance_log_interval == 0:
                    trainer.log_performance_stats()
        
        trainer.add_callback(PerformanceCallback())
    
    # Start training
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully")
        
        # Save final performance statistics
        if improved_config.save_performance_stats:
            import json
            stats = trainer.get_performance_stats()
            stats_path = Path(output_dir) / "performance_stats.json"
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Performance statistics saved to {stats_path}")
        
        # Clear caches to free memory
        trainer.clear_caches()
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        # Log performance stats even on failure for debugging
        if improved_config.save_performance_stats:
            try:
                stats = trainer.get_performance_stats()
                error_stats_path = Path(output_dir) / "error_performance_stats.json"
                error_stats_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(error_stats_path, 'w') as f:
                    json.dump(stats, f, indent=2)
                logger.info(f"Error performance statistics saved to {error_stats_path}")
            except:
                pass
        raise


if __name__ == "__main__":
    main()