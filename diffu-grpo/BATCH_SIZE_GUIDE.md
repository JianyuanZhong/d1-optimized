# Batch Size Configuration Guide for DeepSpeed Training

## 📋 Overview

All batch sizes in the improved GRPO implementation refer to **per-device batch sizes**, following the standard convention for distributed training with DeepSpeed, Accelerate, and Transformers.

## 🔧 Batch Size Types

### 1. **Training Batch Size**
```python
config.optimization.per_device_train_batch_size = 1  # Per-device
```
- **What it is**: Number of training samples processed per GPU/device per forward pass
- **Global batch size**: `per_device_train_batch_size × num_gpus × gradient_accumulation_steps`
- **DeepSpeed**: Automatically handles distribution across devices

### 2. **Generation Batch Size**
```python
config.optimization.generation_batch_size = 4  # Per-device
```
- **What it is**: Number of samples generated simultaneously per GPU during the generation phase
- **Purpose**: Controls memory usage during inference/generation
- **Impact**: Larger values = faster generation but more memory usage

### 3. **Tensor Operations Batch Size**
```python
config.optimization.tensor_batch_size = 1024  # Per-device
```
- **What it is**: Batch size for internal tensor operations (softmax, cross-entropy, etc.)
- **Purpose**: Memory optimization for large tensor operations
- **Impact**: Larger values = faster computation but more memory usage

## 🌐 DeepSpeed Integration

### Automatic Scaling
```python
# Configuration (per-device)
per_device_train_batch_size = 1
generation_batch_size = 4
num_gpus = 8

# Effective batch sizes
effective_train_batch_size = 1 × 8 = 8  # Global training batch
effective_generation_batch_size = 4 × 8 = 32  # Global generation batch
```

### Memory Considerations
With DeepSpeed ZeRO stages:

**ZeRO-1 (Optimizer State Partitioning)**:
```python
# Can use larger per-device batch sizes
config.optimization.per_device_train_batch_size = 2
config.optimization.generation_batch_size = 8
```

**ZeRO-2 (+ Gradient Partitioning)**:
```python
# Moderate batch sizes
config.optimization.per_device_train_batch_size = 1
config.optimization.generation_batch_size = 4
```

**ZeRO-3 (+ Parameter Partitioning)**:
```python
# Conservative batch sizes
config.optimization.per_device_train_batch_size = 1
config.optimization.generation_batch_size = 2
```

## ⚙️ Configuration Examples

### Example 1: 8×A100 80GB with ZeRO-2
```python
config = ImprovedDiffuGRPOConfig()
config.optimization.per_device_train_batch_size = 2
config.optimization.generation_batch_size = 8
config.optimization.tensor_batch_size = 2048

# Effective global batch sizes:
# Training: 2 × 8 = 16
# Generation: 8 × 8 = 64
```

### Example 2: 4×RTX 4090 24GB with ZeRO-3
```python
config = create_memory_optimized_config()
config.optimization.per_device_train_batch_size = 1
config.optimization.generation_batch_size = 2
config.optimization.tensor_batch_size = 512

# Effective global batch sizes:
# Training: 1 × 4 = 4
# Generation: 2 × 4 = 8
```

### Example 3: Single GPU Development
```python
config = ImprovedDiffuGRPOConfig()
config.optimization.per_device_train_batch_size = 1
config.optimization.generation_batch_size = 2
config.optimization.tensor_batch_size = 256

# Effective global batch sizes:
# Training: 1 × 1 = 1
# Generation: 2 × 1 = 2
```

## 🔄 Environment Variable Overrides

You can override batch sizes via environment variables:

```bash
# Set per-device generation batch size
export DIFFU_BATCH_SIZE=8

# Set per-device training batch size  
export DIFFU_TRAIN_BATCH_SIZE=2

# Set tensor operation batch size
export DIFFU_TENSOR_BATCH_SIZE=1024
```

## 📊 Performance Tuning Guidelines

### For Maximum Speed
```python
# Use larger per-device batch sizes if memory allows
config.optimization.generation_batch_size = 8
config.optimization.tensor_batch_size = 2048
```

### For Memory Efficiency
```python
# Use smaller per-device batch sizes
config.optimization.generation_batch_size = 2
config.optimization.tensor_batch_size = 512
config.optimization.enable_gradient_checkpointing = True
```

### For Balanced Performance
```python
# Moderate batch sizes with optimizations
config.optimization.generation_batch_size = 4
config.optimization.tensor_batch_size = 1024
config.optimization.enable_mixed_precision = True
```

## 🚨 Common Pitfalls

### ❌ Don't Do This
```python
# Setting global batch size directly (incorrect)
total_gpus = 8
global_batch_size = 32
per_device_batch_size = global_batch_size  # Wrong!
```

### ✅ Do This Instead
```python
# Calculate per-device batch size correctly
total_gpus = 8
desired_global_batch_size = 32
per_device_batch_size = desired_global_batch_size // total_gpus  # = 4
config.optimization.generation_batch_size = per_device_batch_size
```

## 🔍 Monitoring Batch Sizes

### Check Effective Batch Sizes
```python
trainer = ImprovedDiffuGRPOTrainer(...)

# Log effective batch sizes
num_processes = trainer.accelerator.num_processes
logger.info(f"Per-device generation batch size: {config.optimization.generation_batch_size}")
logger.info(f"Effective global generation batch size: {config.optimization.generation_batch_size * num_processes}")
```

### Memory Usage Monitoring
```python
# Enable memory profiling to tune batch sizes
config.optimization.enable_profiling = True
config.log_memory_usage = True

# Check memory stats after training
stats = trainer.get_performance_stats()
print(f"Peak memory usage: {stats['memory_manager_stats']}")
```

## 🔧 Troubleshooting

### Out of Memory (OOM) Errors
1. **Reduce generation batch size**:
   ```python
   config.optimization.generation_batch_size = 1
   ```

2. **Enable gradient checkpointing**:
   ```python
   config.optimization.enable_gradient_checkpointing = True
   ```

3. **Reduce tensor batch size**:
   ```python
   config.optimization.tensor_batch_size = 256
   ```

### Slow Training
1. **Increase batch sizes if memory allows**:
   ```python
   config.optimization.generation_batch_size = 8
   ```

2. **Enable mixed precision**:
   ```python
   config.optimization.enable_mixed_precision = True
   ```

3. **Use speed-optimized configuration**:
   ```bash
   ./improved_run.sh dataset speed
   ```

## 📚 Additional Resources

- [DeepSpeed Configuration Guide](https://www.deepspeed.ai/getting-started/)
- [Accelerate Distributed Training](https://huggingface.co/docs/accelerate/index)
- [Transformers Training Arguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)

## 💡 Key Takeaways

1. **All batch sizes are per-device** in the improved implementation
2. **Global batch size = per_device_batch_size × num_gpus × gradient_accumulation_steps**
3. **Start with conservative batch sizes** and increase based on available memory
4. **Use environment variables** for easy experimentation
5. **Monitor memory usage** to optimize batch sizes for your hardware