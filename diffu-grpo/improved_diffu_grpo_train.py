import torch
import wandb
import logging
import sys
import yaml
import os
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from trl import ModelConfig
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

# Global logger will be set up later
logger = None


def setup_logging(output_dir: str, log_level: str = "INFO") -> logging.Logger:
    """Set up structured logging with both console and file output."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("diffu_grpo")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplication
    logger.handlers.clear()
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = Path(output_dir) / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized - Console: {console_handler.level}, File: {file_handler.level}")
    logger.info(f"Log file: {log_file}")
    
    return logger


def setup_wandb(config: ImprovedDiffuGRPOConfig, output_dir: str, exp_name: str = None) -> None:
    """Initialize wandb with proper configuration."""
    try:
        # Generate run name with priority: exp_name > config.run_name > default
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if exp_name:
            run_name = f"{exp_name}_{timestamp}"
        else:
            run_name = getattr(config, 'run_name', None) or f"{config.dataset}_{config.model_path.split('/')[-1]}_{timestamp}"
        
        # Initialize wandb
        wandb.init(
            project="diffu-grpo",
            name=run_name,
            config={
                "experiment_name": exp_name,
                "model_path": config.model_path,
                "dataset": config.dataset,
                "num_iterations": config.num_iterations,
                "max_completion_length": config.max_completion_length,
                "max_prompt_length": config.max_prompt_length,
                "num_generations": config.num_generations,
                "mixed_precision": config.optimization.enable_mixed_precision,
                "generation_batch_size": config.optimization.generation_batch_size,
                "per_device_train_batch_size": config.optimization.per_device_train_batch_size,
                "diffusion_steps": config.generation.steps,
                "temperature": config.generation.temperature,
                "cfg_scale": config.generation.cfg_scale,
                "use_new_rewards": config.reward.use_new_rewards,
                "output_dir": output_dir,
            },
            tags=[config.dataset, "diffu-grpo", "training"] + ([exp_name] if exp_name else []),
            dir=output_dir,
        )
        
        logger.info(f"wandb initialized - Project: diffu-grpo, Run: {run_name}")
        if exp_name:
            logger.info(f"  └─ Custom experiment name: {exp_name}")
        logger.info(f"wandb dashboard: {wandb.run.url}")
        
    except Exception as e:
        logger.error(f"Failed to initialize wandb: {str(e)}")
        logger.warning("Continuing without wandb logging")


def setup_model_and_tokenizer(config: ImprovedDiffuGRPOConfig, model_config: ModelConfig, logger: logging.Logger):
    """Set up model and tokenizer with optimized configuration."""
    logger.info("Setting up model and tokenizer")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model with optimizations
    logger.info(f"Loading model: {config.model_path}")
    try:
        model = AutoModel.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
    
    # Load tokenizer
    logger.info("Loading tokenizer")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {str(e)}")
        raise
    
    # Optimize model configuration
    model.config.use_cache = False
    if config.optimization.enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
    
    return model, tokenizer


def setup_peft_config(model_config: ModelConfig, logger: logging.Logger):
    """Set up PEFT configuration for efficient fine-tuning."""
    logger.info("Setting up PEFT configuration")
    
    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=model_config.lora_dropout,
    )
    
    logger.info(f"PEFT config - r: {model_config.lora_r}, alpha: {model_config.lora_alpha}, dropout: {model_config.lora_dropout}")
    return peft_config


def load_dataset(config: ImprovedDiffuGRPOConfig, logger: logging.Logger):
    """Load and prepare dataset based on configuration."""
    logger.info(f"Loading dataset: {config.dataset}")
    
    try:
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
        
        logger.info(f"Dataset loaded successfully - {len(dataset)} samples")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise
    
    # Shuffle dataset with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=42)
    logger.debug("Dataset shuffled with seed=42")
    
    # Use entire dataset for training (evaluation disabled to save GPU memory)
    train_set = dataset
    eval_set = None
    
    logger.info("Dataset configuration:")
    logger.info(f"  └─ Training samples: {len(train_set):,}")
    logger.info(f"  └─ Evaluation: DISABLED (memory optimization)")
    
    return train_set, eval_set


def setup_reward_functions(config: ImprovedDiffuGRPOConfig, logger: logging.Logger):
    """Set up reward functions based on configuration."""
    logger.info("Setting up reward functions")
    
    if config.reward.use_new_rewards:
        logger.info("Using new modular reward functions")
        reward_functions = create_reward_functions(
            config.dataset,
            enable_logging=config.reward.enable_logging,
            log_probability=config.reward.log_probability
        )
        
        # Apply additional configuration settings
        for reward_func in reward_functions:
            # Update reward values if specified
            if hasattr(reward_func, 'correct_reward'):
                reward_func.correct_reward = config.reward.correct_reward
            if hasattr(reward_func, 'reward_value'):
                reward_func.reward_value = config.reward.format_reward
        
        logger.info(f"Configured {len(reward_functions)} reward functions")
        logger.info(f"  └─ Logging enabled: {config.reward.enable_logging}")
        logger.info(f"  └─ Log probability: {config.reward.log_probability}")
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
            boxed_format_reward_func,
        )
        
        if config.dataset == "gsm8k":
            reward_functions = [
                soft_format_reward_func,
                strict_format_reward_func,
                boxed_format_reward_func,
                correctness_reward_func,
            ]
        elif config.dataset == "countdown":
            reward_functions = [countdown_reward_func]
        elif config.dataset == "sudoku":
            reward_functions = [sudoku_reward_func]
        elif config.dataset == "math":
            reward_functions = [
                correctness_reward_func_math,
                boxed_format_reward_func,
            ]
        
        logger.info(f"Configured {len(reward_functions)} legacy reward functions")
        return reward_functions


def create_trainer_config(config: ImprovedDiffuGRPOConfig, model_config: ModelConfig, output_dir: str, dataset_size: int, logger: logging.Logger):
    """Create trainer configuration from improved config."""
    logger.info("Creating trainer configuration")
    
    # Convert our config to the format expected by the trainer
    from diffu_grpo_config import DiffuGRPOConfig
    
    # Calculate training steps to ensure full dataset coverage
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_batch_size = config.optimization.per_device_train_batch_size * num_gpus * 1
    
    # Calculate steps per epoch and total steps
    steps_per_epoch = max(1, dataset_size // effective_batch_size)
    num_epochs = getattr(config, 'num_epochs', 3)
    total_steps = steps_per_epoch * num_epochs
    
    # Ensure minimum training for small datasets
    total_steps = max(total_steps, 100)
    
    # Calculate warmup steps (10% of total steps)
    warmup_steps = max(10, int(0.1 * total_steps))
    
    logger.info("Training schedule:")
    logger.info(f"  ├─ Dataset size: {dataset_size:,}")
    logger.info(f"  ├─ Effective batch size: {effective_batch_size} (per_device: {config.optimization.per_device_train_batch_size} × {num_gpus} GPUs)")
    logger.info(f"  ├─ Steps per epoch: {steps_per_epoch:,}")
    logger.info(f"  ├─ Number of epochs: {num_epochs}")
    logger.info(f"  ├─ Total training steps: {total_steps:,}")
    logger.info(f"  └─ Warmup steps: {warmup_steps:,}")
    
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
        
        # Advanced loss parameters (if using NELBO) - ensure consistency with masking
        trajectory_samples=getattr(config, 'trajectory_samples', 4),
        alpha_schedule=getattr(config, 'alpha_schedule', 'cosine'),
        alpha_min=getattr(config, 'alpha_min', 0.01),
        alpha_max=getattr(config, 'alpha_max', 0.99),
        prior_type=getattr(config, 'prior_type', 'uniform'),
        vocab_size=getattr(config, 'vocab_size', 32000),  # Will be overridden with actual tokenizer size
        
        # Ensure diffusion_steps matches max_timesteps for consistency
        max_timesteps=config.generation.steps,  # Use generation steps as max timesteps
        
        # Optimization settings
        fp16=config.optimization.enable_mixed_precision and not torch.cuda.is_bf16_supported(),
        bf16=config.optimization.enable_mixed_precision and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=config.optimization.enable_gradient_checkpointing,
        
        # Output settings
        output_dir=output_dir,
        logging_steps=min(50, steps_per_epoch // 10),  # Log 10 times per epoch or every 50 steps
        save_steps=max(100, steps_per_epoch // 2),     # Save twice per epoch or every 100 steps
        # eval_steps=max(100, steps_per_epoch // 2),     # Evaluate twice per epoch - DISABLED
        
        # Training duration - use either max_steps OR num_train_epochs, not both
        max_steps=total_steps,
        # num_train_epochs=num_epochs,  # Comment out to use max_steps instead
        warmup_steps=warmup_steps,
        
        # Standard training settings (per-device for distributed training)
        per_device_train_batch_size=config.optimization.per_device_train_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=1e-6,
        remove_unused_columns=False,
        
        # DataLoader settings to ensure full dataset iteration
        dataloader_drop_last=False,  # Don't drop the last incomplete batch
        dataloader_num_workers=4,    # Parallel data loading
        
        # Evaluation and logging - EVALUATION DISABLED TO SAVE GPU MEMORY
        evaluation_strategy="no",
        # load_best_model_at_end=True,  # Disabled since no evaluation
        # metric_for_best_model="eval_loss",  # Disabled since no evaluation
        # greater_is_better=False,  # Disabled since no evaluation
        
        report_to=["wandb"] if wandb.run is not None else [],
    )
    
    return trainer_config


def load_yaml_config(yaml_path: str, logger: logging.Logger) -> ImprovedDiffuGRPOConfig:
    """Load configuration from YAML file."""
    logger.info(f"Loading config from {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create config object from YAML data
    config = ImprovedDiffuGRPOConfig()
    
    # Basic settings
    config.model_path = config_dict.get('model_path', config.model_path)
    config.dataset = config_dict.get('dataset', config.dataset)
    config.num_iterations = config_dict.get('num_iterations', config.num_iterations)
    
    # Handle run_name, output_dir, and training configuration
    if 'run_name' in config_dict:
        config.run_name = config_dict['run_name']
    if 'output_dir' in config_dict:
        config.output_dir = config_dict['output_dir']
    if 'num_epochs' in config_dict:
        config.num_epochs = config_dict['num_epochs']
    
    # Optimization settings
    if 'optimization' in config_dict:
        opt = config_dict['optimization']
        config.optimization.enable_mixed_precision = opt.get('enable_mixed_precision', config.optimization.enable_mixed_precision)
        config.optimization.enable_profiling = opt.get('enable_profiling', config.optimization.enable_profiling)
        config.optimization.generation_batch_size = opt.get('generation_batch_size', config.optimization.generation_batch_size)
        config.optimization.per_device_train_batch_size = opt.get('per_device_train_batch_size', config.optimization.per_device_train_batch_size)
        
        # Memory management settings
        config.memory_cache_interval = opt.get('memory_cache_interval', 50)  # Clear cache every 50 ops instead of every op
    
    # Generation settings
    if 'generation' in config_dict:
        gen = config_dict['generation']
        config.generation.steps = gen.get('steps', config.generation.steps)
    
    # Masking settings
    if 'masking' in config_dict:
        mask = config_dict['masking']
        config.masking.p_mask_prompt = mask.get('p_mask_prompt', config.masking.p_mask_prompt)
        config.masking.strategy_type = mask.get('strategy_type', config.masking.strategy_type)
    
    # Loss settings
    if 'loss' in config_dict:
        loss = config_dict['loss']
        config.loss.adaptive_loss = loss.get('adaptive_loss', config.loss.adaptive_loss)
        config.loss.monte_carlo_loss = loss.get('monte_carlo_loss', False)
        config.loss.hybrid_loss = loss.get('hybrid_loss', False)
        config.loss.trajectory_nelbo_loss = loss.get('trajectory_nelbo_loss', False)
        
        # NELBO-specific parameters
        if config.loss.trajectory_nelbo_loss:
            config.trajectory_samples = loss.get('trajectory_samples', 4)
            config.alpha_schedule = loss.get('alpha_schedule', 'cosine')
            config.alpha_min = loss.get('alpha_min', 0.01)
            config.alpha_max = loss.get('alpha_max', 0.99)
            config.prior_type = loss.get('prior_type', 'uniform')
            config.vocab_size = loss.get('vocab_size', 32000)
    
    # Reward settings
    if 'reward' in config_dict:
        reward = config_dict['reward']
        config.reward.use_new_rewards = reward.get('use_new_rewards', config.reward.use_new_rewards)
        config.reward.enable_logging = reward.get('enable_logging', config.reward.enable_logging)
        config.reward.log_probability = reward.get('log_probability', config.reward.log_probability)
    
    # Override with environment variables if present
    if 'DIFFU_MODEL_PATH' in os.environ:
        config.model_path = os.environ['DIFFU_MODEL_PATH']
    if 'DIFFU_DATASET' in os.environ:
        config.dataset = os.environ['DIFFU_DATASET']
    if 'DIFFU_NUM_ITERATIONS' in os.environ:
        config.num_iterations = int(os.environ['DIFFU_NUM_ITERATIONS'])
    if 'DIFFU_ENABLE_PROFILING' in os.environ:
        config.optimization.enable_profiling = os.environ['DIFFU_ENABLE_PROFILING'].lower() == 'true'
    
    return config


def main():
    """Main training function with improved architecture."""
    # Parse command line arguments to get config file path
    import argparse
    parser = argparse.ArgumentParser(description='Improved Diffusion GRPO Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--exp-name', type=str, default=None, 
                        help='Custom experiment name for wandb (will be suffixed with timestamp)')
    args = parser.parse_args()
    
    # Set output directory early for logging setup
    # We'll read a minimal config first to determine output dir
    try:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        output_dir = config_dict.get('output_dir', 'checkpoints/default')
    except Exception:
        output_dir = 'checkpoints/default'
    
    # Set up logging system
    global logger
    logger = setup_logging(output_dir, args.log_level)
    
    logger.info("=" * 80)
    logger.info("🚀 Starting Improved Diffusion GRPO Training")
    logger.info("=" * 80)
    
    # Load configuration from YAML file
    try:
        improved_config = load_yaml_config(args.config, logger)
        logger.info(f"✓ Successfully loaded config from {args.config}")
    except Exception as e:
        logger.error(f"✗ Failed to load config from {args.config}: {str(e)}")
        logger.warning("Using default GSM8K config")
        improved_config = create_gsm8k_config()
    
    # Update output directory from config if available
    if hasattr(improved_config, 'output_dir') and improved_config.output_dir:
        output_dir = improved_config.output_dir
    else:
        output_dir = f"checkpoints/{improved_config.dataset}_improved_{improved_config.optimization.generation_batch_size}"
    
    # Initialize wandb
    setup_wandb(improved_config, output_dir, args.exp_name)
    
    # Create model config
    model_config = ModelConfig()
    
    # Set random seed for reproducibility
    set_random_seed(42)
    logger.info("✓ Random seed set to 42")
    
    # Log main configuration
    logger.info("📋 Configuration Summary:")
    logger.info(f"  ├─ Model: {improved_config.model_path}")
    logger.info(f"  ├─ Dataset: {improved_config.dataset}")
    logger.info(f"  ├─ Iterations: {improved_config.num_iterations}")
    logger.info(f"  ├─ Mixed precision: {improved_config.optimization.enable_mixed_precision}")
    logger.info(f"  ├─ Generation batch size: {improved_config.optimization.generation_batch_size}")
    logger.info(f"  ├─ Per-device batch size: {improved_config.optimization.per_device_train_batch_size}")
    
    # Log memory optimization settings
    memory_interval = getattr(improved_config, 'memory_cache_interval', 100)
    logger.info(f"  ├─ Memory cache interval: {memory_interval}")
    logger.info(f"  └─ Output directory: {output_dir}")
    
    # Set up model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(improved_config, model_config, logger)
    
    # Infer vocabulary size from tokenizer for NELBO loss
    if getattr(improved_config.loss, 'trajectory_nelbo_loss', False):
        vocab_size = len(tokenizer)
        logger.info(f"📚 Inferred vocabulary size from tokenizer: {vocab_size:,}")
        # Update config with actual vocab size
        improved_config.vocab_size = vocab_size
    
    # Load dataset
    train_dataset, eval_dataset = load_dataset(improved_config, logger)
    
    # Set up reward functions
    reward_functions = setup_reward_functions(improved_config, logger)
    
    # Set up PEFT configuration
    peft_config = setup_peft_config(model_config, logger)
    
    # Create trainer configuration (after vocab size is inferred)
    trainer_config = create_trainer_config(improved_config, model_config, output_dir, len(train_dataset), logger)
    
    # Initialize improved trainer
    logger.info("🔧 Initializing GRPO trainer")
    trainer = ImprovedDiffuGRPOTrainer(
        args=trainer_config,
        model=model,
        peft_config=peft_config,
        reward_funcs=reward_functions,
        train_dataset=train_dataset,
        eval_dataset=None,  # Disabled to save GPU memory
        processing_class=tokenizer,
        enable_profiling=improved_config.optimization.enable_profiling,
        masking_strategy=improved_config.masking.strategy_type,
        adaptive_loss=improved_config.loss.adaptive_loss,
        monte_carlo_loss=getattr(improved_config.loss, 'monte_carlo_loss', False),
        hybrid_loss=getattr(improved_config.loss, 'hybrid_loss', False),
        trajectory_nelbo_loss=getattr(improved_config.loss, 'trajectory_nelbo_loss', False),
    )
    
    # Update vocabulary size from tokenizer if using TrajectoryNELBOLoss
    if getattr(improved_config.loss, 'trajectory_nelbo_loss', False):
        trainer.update_vocab_size_from_tokenizer(tokenizer)
        logger.info("✓ TrajectoryNELBOLoss vocabulary size updated from tokenizer")
    
    # Set up performance monitoring
    if getattr(improved_config, 'enable_performance_logging', False):
        logger.info("📊 Performance monitoring enabled")
        
        from transformers import TrainerCallback
        
        class PerformanceCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step % getattr(improved_config, 'performance_log_interval', 100) == 0:
                    trainer.log_performance_stats()
        
        trainer.add_callback(PerformanceCallback())
    
    # Start training
    logger.info("🏃 Starting training...")
    logger.info("=" * 80)
    
    try:
        trainer.train()
        logger.info("=" * 80)
        logger.info("🎉 Training completed successfully!")
        
        # Save the final model and tokenizer
        logger.info("💾 Saving final model and tokenizer...")
        final_model_path = Path(output_dir) / "final_model"
        final_model_path.mkdir(parents=True, exist_ok=True)
        
        # Save the adapter weights (LoRA)
        trainer.model.save_pretrained(final_model_path)
        logger.info(f"  ✓ Adapter weights saved to {final_model_path}")
        
        # Save the tokenizer
        tokenizer.save_pretrained(final_model_path)
        logger.info(f"  ✓ Tokenizer saved to {final_model_path}")
        
        # Optionally save the full merged model (adapter + base model)
        try:
            logger.info("🔄 Merging and saving full model...")
            merged_model_path = Path(output_dir) / "merged_model"
            merged_model_path.mkdir(parents=True, exist_ok=True)
            
            # Get the base model and merge with adapter
            base_model = trainer.model.get_base_model()
            merged_model = trainer.model.merge_and_unload()
            
            # Save merged model
            merged_model.save_pretrained(merged_model_path)
            tokenizer.save_pretrained(merged_model_path)
            logger.info(f"  ✓ Full merged model saved to {merged_model_path}")
            
        except Exception as e:
            logger.warning(f"  ⚠️ Could not save merged model: {str(e)}")
            logger.info("  → Adapter weights are still saved and can be loaded with the base model")
        
        # Save final performance statistics
        if getattr(improved_config, 'save_performance_stats', False):
            try:
                import json
                stats = trainer.get_performance_stats()
                stats_path = Path(output_dir) / "performance_stats.json"
                stats_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(stats_path, 'w') as f:
                    json.dump(stats, f, indent=2)
                logger.info(f"  ✓ Performance statistics saved to {stats_path}")
            except Exception as e:
                logger.warning(f"  ⚠️ Could not save performance stats: {str(e)}")
        
        # Clear caches to free memory
        trainer.clear_caches()
        logger.info("  ✓ Memory caches cleared")
        
        # Clean up wandb
        try:
            if wandb.run is not None:
                wandb.finish()
                logger.info("  ✓ wandb run completed")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not finish wandb run: {str(e)}")
        
        logger.info("=" * 80)
        logger.info("🎊 All tasks completed successfully!")
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"💥 Training failed: {str(e)}")
        logger.error("=" * 80)
        
        # Log performance stats even on failure for debugging
        if getattr(improved_config, 'save_performance_stats', False):
            try:
                import json
                stats = trainer.get_performance_stats()
                error_stats_path = Path(output_dir) / "error_performance_stats.json"
                error_stats_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(error_stats_path, 'w') as f:
                    json.dump(stats, f, indent=2)
                logger.info(f"📊 Error performance statistics saved to {error_stats_path}")
            except Exception as stats_error:
                logger.warning(f"Could not save error performance stats: {str(stats_error)}")
        
        # Clean up wandb
        try:
            if wandb.run is not None:
                wandb.finish(exit_code=1)
        except:
            pass
        
        raise


if __name__ == "__main__":
    main()