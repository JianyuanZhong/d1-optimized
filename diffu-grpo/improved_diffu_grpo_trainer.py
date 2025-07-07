import torch
from trl.trainer.grpo_trainer import GRPOTrainer
from typing import Any, Callable, Optional, Union, Sized, Dict, List
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback, Trainer
from datasets import Dataset, IterableDataset
import warnings
import torch.nn.functional as F
from trl.trainer.grpo_config import GRPOConfig
from trl.extras.profiling import profiling_decorator, profiling_context
from transformers.utils import is_peft_available
from torch import nn
from trl.import_utils import is_rich_available, is_vllm_available
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)
import wandb
import pandas as pd

# Import our new modular components
from core.memory_manager import MemoryManager
from core.diffusion_generator import DiffusionGenerator
from core.masking_strategy import create_masking_strategy, DiffusionMaskingStrategy
from core.grpo_loss import GRPOLoss, AdaptiveGRPOLoss, MonteCarloGRPOLoss, HybridGRPOLoss
from core.reward_functions import RewardFunctionManager, create_reward_functions
from core.tensor_ops import TensorOpsOptimizer, MemoryEfficientOperations

if is_peft_available():
    from peft import PeftConfig, get_peft_model

import logging

logger = logging.getLogger(__name__)

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class ImprovedDiffuGRPOTrainer(GRPOTrainer):
    """
    Improved Group Relative Policy Optimization (GRPO) Trainer for Diffusion Language Models.
    
    This class uses a modular architecture with optimized components for better performance,
    maintainability, and memory efficiency.
    
    Key improvements:
    - Modular architecture with separated concerns
    - Optimized memory management with context managers
    - Efficient tensor operations and batching
    - Better error handling and logging
    - Caching for expensive operations
    - Support for adaptive hyperparameters
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        peft_config: Optional["PeftConfig"] = None,
        enable_profiling: bool = False,
        masking_strategy: str = "diffusion",
        adaptive_loss: bool = False,
        monte_carlo_loss: bool = False,
        hybrid_loss: bool = False,
    ):
        # Ensure beta is never 0 - override parent class setting
        self.beta = max(getattr(args, 'beta', 0.04), 0.01)
        logger.info(f"Beta coefficient set to: {self.beta} (ensured non-zero)")
        # Initialize the parent class
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        
        # Initialize our modular components with optimized memory management
        # For Monte Carlo operations, we need more frequent cache clearing
        cache_clear_interval = getattr(args, 'memory_cache_interval', 20)  # More frequent for MC operations
        self.memory_manager = MemoryManager(
            enable_profiling=enable_profiling, 
            cache_clear_interval=cache_clear_interval
        )
        self.tensor_ops = TensorOpsOptimizer(device=self.accelerator.device)
        
        # Initialize masking strategy
        self.masking_strategy = create_masking_strategy(
            masking_strategy,
            p_mask_prompt=getattr(args, 'p_mask_prompt', 0.3),
            cache_size=1000
        )
        
        # Initialize diffusion generator
        self.diffusion_generator = DiffusionGenerator(
            memory_manager=self.memory_manager,
            enable_mixed_precision=getattr(args, 'fp16', False) or getattr(args, 'bf16', False)
        )
        
        # Initialize GRPO loss with Monte Carlo support
        loss_kwargs = {
            'masking_strategy': self.masking_strategy,
            'memory_manager': self.memory_manager,
            'epsilon': getattr(args, 'epsilon', 0.2),
            'beta': max(getattr(args, 'beta', 0.04), 0.01),  # Ensure beta is never 0, minimum 0.01
            'enable_mixed_precision': getattr(args, 'fp16', False) or getattr(args, 'bf16', False)
        }
        
        # Add Monte Carlo specific parameters
        if monte_carlo_loss or hybrid_loss:
            loss_kwargs.update({
                'mc_num': getattr(args, 'mc_num', 128),
                'mc_batch_size': getattr(args, 'mc_batch_size', 16),
                'cfg_scale': getattr(args, 'cfg_scale', 0.0),
                'mask_id': getattr(args, 'mask_id', 126336)
            })
        
        # Initialize the appropriate loss class
        if hybrid_loss:
            self.grpo_loss = HybridGRPOLoss(
                use_monte_carlo=True,
                monte_carlo_warmup_steps=getattr(args, 'monte_carlo_warmup_steps', 1000),
                **loss_kwargs
            )
            logger.info("Using HybridGRPOLoss with Monte Carlo estimation")
        elif monte_carlo_loss:
            self.grpo_loss = MonteCarloGRPOLoss(**loss_kwargs)
            logger.info("Using MonteCarloGRPOLoss")
        elif adaptive_loss:
            adaptive_kwargs = {
                'initial_epsilon': getattr(args, 'epsilon', 0.2),
                'initial_beta': max(getattr(args, 'beta', 0.05), 0.01),  # Ensure never 0
                'final_epsilon': getattr(args, 'final_epsilon', 0.1),
                'final_beta': max(getattr(args, 'final_beta', 0.02), 0.01),  # Ensure never 0
                'total_steps': args.max_steps if hasattr(args, 'max_steps') else 10000
            }
            loss_kwargs.update(adaptive_kwargs)
            self.grpo_loss = AdaptiveGRPOLoss(**loss_kwargs)
            logger.info("Using AdaptiveGRPOLoss")
        else:
            self.grpo_loss = GRPOLoss(**loss_kwargs)
            logger.info("Using standard GRPOLoss")
        
        # Initialize reward function manager if using new reward functions
        if hasattr(args, 'dataset') and isinstance(reward_funcs, str):
            try:
                new_reward_funcs = create_reward_functions(args.dataset)
                self.reward_manager = RewardFunctionManager(new_reward_funcs)
                self.use_new_rewards = True
            except:
                self.use_new_rewards = False
                logger.warning("Using legacy reward functions")
        else:
            self.use_new_rewards = False
        
        self.enable_profiling = enable_profiling
        
        # Performance tracking
        self.performance_stats = {
            'generation_time': [],
            'loss_computation_time': [],
            'memory_usage': [],
            'cache_hit_rates': []
        }
        
        # Initialize metrics tracking
        self._metrics = {
            'train': {},
            'eval': {}
        }
        
        # Initialize logging configuration
        self.log_completions = getattr(args, 'log_completions', True)

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss using the modular GRPO loss component."""
        if return_outputs:
            raise ValueError("The ImprovedDiffuGRPOTrainer does not support returning outputs")
        
        # Get current iteration index
        iteration_idx = self._step % self.args.num_iterations
        
        # Update adaptive components if needed
        if hasattr(self.grpo_loss, 'update_step'):
            self.grpo_loss.update_step(self.state.global_step)
        
        if hasattr(self.masking_strategy, 'update_step'):
            self.masking_strategy.update_step(self.state.global_step)
        
        # Compute loss using modular component
        with self.memory_manager.managed_forward("compute_loss"):
            loss, metrics = self.grpo_loss.compute_loss(
                model, inputs, iteration_idx, self.args.num_iterations
            )
        
        # Log metrics
        mode = "eval" if self.control.should_evaluate else "train"
        for metric_name, metric_value in metrics.items():
            if metric_name not in self._metrics[mode]:
                self._metrics[mode][metric_name] = []
            
            # Ensure metric_value is a valid scalar and create tensor on correct device
            if metric_value is not None and not (isinstance(metric_value, float) and np.isnan(metric_value)):
                metric_tensor = torch.tensor(float(metric_value), device=self.accelerator.device, dtype=torch.float32)
                gathered_metric = self.accelerator.gather_for_metrics(metric_tensor).mean().item()
                self._metrics[mode][metric_name].append(gathered_metric)
            else:
                # Handle None or NaN values
                self._metrics[mode][metric_name].append(0.0)
        
        return loss

    def generate_optimized(
        self,
        model,
        prompt,
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
        batch_size=None,
    ):
        """Use the optimized diffusion generator."""
        return self.diffusion_generator.generate(
            model=model,
            prompt=prompt,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            mask_id=mask_id,
            batch_size=batch_size or getattr(self.args, 'generation_batch_size', 4)
        )

    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Prepare inputs with optimized memory management."""
        mode = "eval" if self.control.should_evaluate else "train"
        
        with self.memory_manager.managed_forward("prepare_inputs"):
            if mode == "train":
                if self.state.global_step % self.num_iterations == 0:
                    inputs = self._generate_and_score_completions_optimized(inputs)
                    self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
                else:
                    inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
                self._step += 1
            else:
                inputs = self._generate_and_score_completions_optimized(inputs)
        
        return inputs

    def _generate_and_score_completions_optimized(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Optimized generation and scoring with better memory management."""
        device = self.accelerator.device

        # Extract prompts and prepare inputs
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs
        ]
        
        with self.memory_manager.managed_tensor_ops([], "prompt_processing"):
            prompt_inputs = self.processing_class(
                text=prompts_text,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
            prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

            if self.max_prompt_length is not None:
                prompt_ids = prompt_ids[:, -self.max_prompt_length :]
                prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generation configuration
        gen_config = {
            'gen_length': self.args.max_completion_length,
            'block_length': getattr(self.args, 'block_length', 64),
            'steps': getattr(self.args, 'diffusion_steps', 64),
            'temperature': getattr(self.args, 'temperature', 0.0),
            'cfg_scale': getattr(self.args, 'cfg_scale', 0.0),
            'remasking': getattr(self.args, 'remasking', 'low_confidence'),
            'mask_id': getattr(self.args, 'mask_id', 126336),
        }

        # Generate completions using optimized generator
        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            with self.memory_manager.managed_generation(prompt_ids.size(0), gen_config['gen_length']):
                prompt_completion_ids = self.generate_optimized(
                    model=unwrapped_model,
                    prompt=prompt_ids,
                    batch_size=getattr(self.args, 'generation_batch_size', 4),
                    **gen_config
                )

        # Process generated completions
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Create completion mask efficiently
        completion_mask = self._create_completion_mask_optimized(
            completion_ids, self.processing_class.eos_token_id, device
        )
        
        logits_to_keep = completion_ids.size(1)
        
        # Generate mask seeds
        if getattr(self.args, 'random_masking', True):
            mask_seeds = torch.randint(0, 2**12, (self.num_iterations,), device=device)
        else:
            mask_seeds = torch.tensor([42] * self.num_iterations, device=device)

        # Compute log probabilities efficiently
        all_old_per_token_logps = None
        all_ref_per_token_logps = None
        
        # Always create expanded tensor when needed for logp computation
        prompt_completion_ids_expanded = prompt_completion_ids.unsqueeze(0).expand(
            self.num_iterations, -1, -1
        )
        
        with torch.no_grad():
            if self.num_iterations > 1:
                # For old logprobs, we should use the current model, not ref_model
                with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
                    all_old_per_token_logps = self.grpo_loss._compute_per_token_logps(
                        unwrapped_model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds.tolist()
                    )

            if self.beta > 0.0:
                if self.ref_model is not None:
                    all_ref_per_token_logps = self.grpo_loss._compute_per_token_logps(
                        self.ref_model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds.tolist()
                    )
                else:
                    # When ref_model is None but PEFT is used, use model with disabled adapter
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
                            all_ref_per_token_logps = self.grpo_loss._compute_per_token_logps(
                                unwrapped_model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds.tolist()
                            )

        # Process completions for reward computation
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Compute rewards
        rewards, advantages = self._compute_rewards_and_advantages_optimized(
            prompts, completions, inputs, device
        )

        # Log rewards and completions
        rewards_per_func = None
        if hasattr(self, '_last_rewards_per_func'):
            rewards_per_func = self._last_rewards_per_func
        self._log_rewards_and_completions(
            prompts, completions, rewards, rewards_per_func, completion_mask, mode="train"
        )

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": all_old_per_token_logps,
            "ref_per_token_logps": all_ref_per_token_logps,
            "advantages": advantages,
            "mask_seeds": mask_seeds,
        }

    def _create_completion_mask_optimized(self, completion_ids: torch.Tensor, 
                                        eos_token_id: int, device: torch.device) -> torch.Tensor:
        """Create completion mask with optimized operations."""
        is_eos = completion_ids == eos_token_id
        
        # Use optimized operations
        batch_size, seq_len = completion_ids.shape
        eos_idx = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
        
        # Find first EOS token efficiently
        has_eos = is_eos.any(dim=1)
        if has_eos.any():
            eos_positions = is_eos.int().argmax(dim=1)
            eos_idx[has_eos] = eos_positions[has_eos]
        
        # Create sequence indices
        sequence_indices = torch.arange(seq_len, device=device).expand(batch_size, -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        return completion_mask

    def _compute_rewards_and_advantages_optimized(self, prompts, completions, inputs, device):
        """Compute rewards and advantages with optimized operations."""
        if self.use_new_rewards:
            # Use new reward function manager
            rewards_list, reward_metadata = self.reward_manager.compute_rewards(
                prompts, completions, **{key: [example[key] for example in inputs] 
                                       for key in inputs[0] if key not in ["prompt", "completion"]}
            )
            rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
        else:
            # Use legacy reward computation
            rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
            
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                with profiling_context(self, f"reward_{i}"):
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    
                    try:
                        output_reward_func = reward_func(
                            prompts=prompts,
                            completions=completions,
                            step=self._step,
                            run_name=self.args.output_dir,
                            **reward_kwargs,
                        )
                        output_reward_func = [
                            reward if reward is not None else torch.nan for reward in output_reward_func
                        ]
                        rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
                    except Exception as e:
                        logger.warning(f"Reward function {i} failed: {str(e)}")
                        rewards_per_func[:, i] = 0.0

            rewards_per_func = gather(rewards_per_func)
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            
            # Store for logging
            self._last_rewards_per_func = rewards_per_func

        # Compute advantages efficiently
        grouped_rewards = rewards.view(-1, self.num_generations)
        mean_grouped_rewards = grouped_rewards.mean(dim=1)
        std_grouped_rewards = grouped_rewards.std(dim=1)
        
        # Expand means for advantage computation
        expanded_means = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - expanded_means
        
        # Get slice for current process
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        
        return rewards, advantages

    def _log_rewards_and_completions(self, prompts, completions, rewards, rewards_per_func=None, completion_mask=None, mode="train"):
        """Log rewards and completions to console and wandb."""
        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            # Gather data for logging
            prompts_text = prompts if isinstance(prompts[0], str) else [p[0]["content"] for p in prompts]
            completions_text = completions if isinstance(completions[0], str) else [c[0]["content"] for c in completions]
            
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()
            
            # Log reward function metrics
            if rewards_per_func is not None:
                for i, reward_func in enumerate(self.reward_funcs):
                    if isinstance(reward_func, nn.Module):
                        reward_func_name = reward_func.config._name_or_path.split("/")[-1]
                    else:
                        reward_func_name = reward_func.__name__
                    mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
                    if mode not in self._metrics:
                        self._metrics[mode] = {}
                    if f"rewards/{reward_func_name}" not in self._metrics[mode]:
                        self._metrics[mode][f"rewards/{reward_func_name}"] = []
                    self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
            
            # Log completion metrics
            if completion_mask is not None:
                completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
                if "completion_length" not in self._metrics[mode]:
                    self._metrics[mode]["completion_length"] = []
                self._metrics[mode]["completion_length"].append(completion_length)
            
            # Log reward statistics
            if "reward" not in self._metrics[mode]:
                self._metrics[mode]["reward"] = []
            self._metrics[mode]["reward"].append(rewards.mean().item())
            
            if "reward_std" not in self._metrics[mode]:
                self._metrics[mode]["reward_std"] = []
            grouped_rewards = rewards.view(-1, self.num_generations)
            std_grouped_rewards = grouped_rewards.std(dim=1)
            self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
            
            # Zero std ratio
            zero_std_ratio = (std_grouped_rewards == 0).float().mean().item()
            if "zero_std_ratio" not in self._metrics[mode]:
                self._metrics[mode]["zero_std_ratio"] = []
            self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)
            
            if self.accelerator.is_main_process:
                # Console logging
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                
                # wandb logging
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})
                    
                    # Log metrics to wandb
                    wandb_metrics = {}
                    for metric_name, values in self._metrics[mode].items():
                        if values:
                            wandb_metrics[f"{mode}/{metric_name}"] = values[-1]
                    if wandb_metrics:
                        wandb.log(wandb_metrics)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'trainer_stats': self.performance_stats,
            'memory_manager_stats': self.memory_manager.get_memory_stats(),
            'generator_stats': self.diffusion_generator.get_generation_stats(),
            'loss_stats': self.grpo_loss.get_performance_stats(),
            'tensor_ops_cache_size': len(self.tensor_ops._shape_cache)
        }
        
        if hasattr(self.masking_strategy, 'get_cache_stats'):
            stats['masking_stats'] = self.masking_strategy.get_cache_stats()
        
        if self.use_new_rewards:
            stats['reward_stats'] = self.reward_manager.get_stats()
        
        return stats

    def clear_caches(self):
        """Clear all caches to free memory."""
        self.memory_manager.clear_stats()
        self.diffusion_generator.clear_cache()
        self.tensor_ops.clear_cache()
        
        if hasattr(self.masking_strategy, 'clear_cache'):
            self.masking_strategy.clear_cache()

    def log_performance_stats(self):
        """Log performance statistics."""
        stats = self.get_performance_stats()
        logger.info("=== Performance Statistics ===")
        
        for component, component_stats in stats.items():
            logger.info(f"{component}: {component_stats}")
        
        # Log to wandb if available
        if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
            wandb.log({"performance_stats": stats}, step=self.state.global_step)