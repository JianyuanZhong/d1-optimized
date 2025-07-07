#!/bin/bash

# Improved run script for diffusion GRPO training
# This script demonstrates how to use the new modular architecture
#
# Usage: ./improved_run.sh [DATASET] [OPTIMIZATION] [EXP_NAME]
#   DATASET: gsm8k, countdown, sudoku, math (default: gsm8k)
#   OPTIMIZATION: speed, memory, balanced, 8gpu (default: speed)
#   EXP_NAME: Custom experiment name for wandb (optional)

export LOGDIR=checkpoints
mkdir -p $LOGDIR

# Configuration
DATASET=${1:-"gsm8k"}  # Default to sudoku, can be: gsm8k, countdown, sudoku, math
OPTIMIZATION=${2:-"speed"}  # Can be: speed, memory, balanced
EXP_NAME=${3:-""}  # Custom experiment name for wandb (optional)
RUN_NAME=${DATASET}_improved_${OPTIMIZATION}
MODEL_PATH=${MODEL_PATH:-"GSAI-ML/LLaDA-8B-Instruct"}
NUM_ITER=${NUM_ITER:-12}
NUM_EPOCHS=${NUM_EPOCHS:-20}

# Set environment variables for configuration
export DIFFU_MODEL_PATH=$MODEL_PATH
export DIFFU_DATASET=$DATASET
export DIFFU_NUM_ITERATIONS=$NUM_ITER
export DIFFU_ENABLE_PROFILING=true

# Select the appropriate accelerate config based on optimization mode
ACCELERATE_CONFIG="accelerate.yaml"

# Optimization-specific settings (all batch sizes are per-device)
case $OPTIMIZATION in
    "speed")
        echo "Using speed optimizations"
        export DIFFU_MIXED_PRECISION=true
        export DIFFU_BATCH_SIZE=6  # Per-device generation batch size
        export DIFFU_TRAIN_BATCH_SIZE=6  # Per-device training batch size
        export DIFFU_ENABLE_GRADIENT_CHECKPOINTING=false
        ACCELERATE_CONFIG="accelerate_8gpu.yaml"  # Use multi-GPU config
        ;;
    "memory")
        echo "Using memory optimizations"
        export DIFFU_MIXED_PRECISION=true
        export DIFFU_BATCH_SIZE=2  # Per-device generation batch size
        export DIFFU_TRAIN_BATCH_SIZE=1  # Per-device training batch size
        ;;
    "balanced")
        echo "Using balanced optimizations"
        export DIFFU_MIXED_PRECISION=true
        export DIFFU_BATCH_SIZE=4  # Per-device generation batch size
        export DIFFU_TRAIN_BATCH_SIZE=1  # Per-device training batch size
        ACCELERATE_CONFIG="accelerate_8gpu.yaml"  # Use multi-GPU config
        ;;
    "8gpu")
        echo "Using 8-GPU optimized settings"
        export DIFFU_MIXED_PRECISION=true
        export DIFFU_BATCH_SIZE=6  # Per-device generation batch size (total: 48)
        export DIFFU_TRAIN_BATCH_SIZE=6  # Per-device training batch size (total: 48)
        export DIFFU_GRADIENT_ACCUMULATION_STEPS=1  # No accumulation needed with 8 GPUs
        export DIFFU_ENABLE_GRADIENT_CHECKPOINTING=true
        ACCELERATE_CONFIG="accelerate_8gpu.yaml"  # Use multi-GPU config
        ;;
    *)
        echo "Unknown optimization mode: $OPTIMIZATION"
        echo "Using default settings"
        ;;
esac

# Dataset-specific settings
case $DATASET in
    "gsm8k")
        export DIFFU_GENERATION_STEPS=128
        export DIFFU_P_MASK_PROMPT=0.3
        ;;
    "countdown")
        export DIFFU_GENERATION_STEPS=32
        export DIFFU_P_MASK_PROMPT=0.2
        ;;
    "sudoku")
        export DIFFU_GENERATION_STEPS=32
        export DIFFU_P_MASK_PROMPT=0.4
        ;;
    "math")
        export DIFFU_GENERATION_STEPS=64
        export DIFFU_P_MASK_PROMPT=0.3
        ;;
esac

echo "Starting improved GRPO training with:"
echo "  Dataset: $DATASET"
echo "  Optimization: $OPTIMIZATION"
echo "  Experiment name: ${EXP_NAME:-"auto-generated"}"
echo "  Model: $MODEL_PATH"
echo "  Iterations: $NUM_ITER"
echo "  Epochs: $NUM_EPOCHS"
echo "  Run name: $RUN_NAME"
echo "  Output dir: checkpoints/$RUN_NAME"
echo "  Accelerate config: $ACCELERATE_CONFIG"

# Create a simple config file for the run
cat > config_${RUN_NAME}.yaml << EOF
model_path: "$MODEL_PATH"
dataset: "$DATASET"
num_iterations: $NUM_ITER
num_epochs: ${NUM_EPOCHS:-3}  # Number of epochs to train for
run_name: "$RUN_NAME"
output_dir: "checkpoints/$RUN_NAME"
optimization:
  enable_mixed_precision: true
  enable_profiling: true
  generation_batch_size: ${DIFFU_BATCH_SIZE:-6}  # Per-device generation batch size
  per_device_train_batch_size: ${DIFFU_TRAIN_BATCH_SIZE:-6}  # Per-device training batch size
generation:
  steps: ${DIFFU_GENERATION_STEPS:-128}
masking:
  p_mask_prompt: ${DIFFU_P_MASK_PROMPT:-0.3}
  strategy_type: "diffusion"
loss:
  adaptive_loss: false
reward:
  use_new_rewards: true
  enable_logging: true
  log_probability: 0.2
EOF

echo "Configuration saved to config_${RUN_NAME}.yaml"

# Run the improved training script
# Build the command with optional experiment name
TRAIN_CMD="accelerate launch \
    --config_file $ACCELERATE_CONFIG \
    --main_process_port 12346 \
    improved_diffu_grpo_train.py \
    --config config_${RUN_NAME}.yaml"

# Add experiment name if provided
if [ -n "$EXP_NAME" ]; then
    TRAIN_CMD="$TRAIN_CMD --exp-name $EXP_NAME"
fi

echo "Running command: $TRAIN_CMD"
eval $TRAIN_CMD

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Results saved to: checkpoints/$RUN_NAME"
    
    # Display performance statistics if available
    if [ -f "checkpoints/$RUN_NAME/performance_stats.json" ]; then
        echo "Performance statistics:"
        python -m json.tool "checkpoints/$RUN_NAME/performance_stats.json" | head -20
    fi
else
    echo "Training failed. Check logs for details."
    exit 1
fi