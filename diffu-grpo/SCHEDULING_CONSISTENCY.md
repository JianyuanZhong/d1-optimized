# Scheduling Consistency Between Rollout Generation and Loss Computation

## 🎯 **Problem Solved**

Previously, the `DiffusionMaskingStrategy` (used during rollout generation) and `TrajectoryNELBOLoss` (used during loss computation) had **inconsistent scheduling**:

- **Rollout stage**: Used simple probability-based masking (`p_mask_prompt = 0.3`)
- **Loss stage**: Used theoretical α-schedule (cosine/linear/exponential)

This mismatch could lead to suboptimal training because the model sees different noise patterns during generation vs. loss computation.

## ✅ **Solution Implemented**

### 1. **Unified Alpha Scheduling**

Both `DiffusionMaskingStrategy` and `TrajectoryNELBOLoss` now use the **same α-schedule**:

```python
# Both classes now share identical schedule creation
def _create_alpha_schedule(self) -> torch.Tensor:
    t = torch.linspace(0, 1, self.max_timesteps + 1)
    
    if self.alpha_schedule == "cosine":
        alpha_t = torch.cos(t * torch.pi / 2) ** 2
        alpha_t = self.alpha_min + (self.alpha_max - self.alpha_min) * alpha_t
    # ... same logic for linear/exponential
```

### 2. **Timestep-Aware Masking**

`DiffusionMaskingStrategy.forward_process()` now accepts a `timestep` parameter:

```python
def forward_process(self, batch, prompt_index, mask_id, seed=None, timestep=None):
    # For completion tokens: use alpha schedule (1 - alpha_t = noise level)
    alpha_t = self.alpha_t[timestep].to(device)
    t_p_completion = torch.full((b,), 1.0 - alpha_t.item(), device=device)
    
    # For prompt tokens: still use configured p_mask_prompt
    t_p_prompt = torch.full((b,), self.p_mask_prompt, device=device)
```

### 3. **Automatic Synchronization**

The trainer automatically synchronizes the schedules:

```python
# In ImprovedDiffuGRPOTrainer.__init__()
if trajectory_nelbo_loss:
    from core.masking_strategy import sync_masking_with_nelbo
    sync_masking_with_nelbo(self.masking_strategy, self.grpo_loss)
```

### 4. **Consistent Parameter Flow**

Parameters flow from config → masking strategy → NELBO loss:

```python
# Config YAML
loss:
  trajectory_nelbo_loss: true
  alpha_schedule: "cosine"
  alpha_min: 0.01
  alpha_max: 0.99

# → Passed to masking strategy initialization
# → Synchronized with NELBO loss
# → Used consistently in both generation and loss computation
```

## 🔄 **How It Works**

### **During Rollout Generation:**
1. `DiffusionGenerator` calls `masking_strategy.forward_process()` 
2. Masking strategy uses α-schedule to determine noise level
3. For timestep `t`: completion tokens masked with probability `1 - α_t`
4. Prompt tokens still use configured `p_mask_prompt`

### **During Loss Computation:**
1. `TrajectoryNELBOLoss` samples trajectory timesteps
2. Uses **same α-schedule** to compute forward process q(x_t|x_0)
3. Consistent noise levels between generation and loss computation

### **Key Benefits:**
- ✅ **Consistent noise patterns** across training pipeline
- ✅ **Theoretical soundness** - NELBO bound is meaningful
- ✅ **Automatic synchronization** - no manual parameter matching needed
- ✅ **Backward compatibility** - fallback for non-timestep strategies

## 📊 **Schedule Comparison**

```python
# Example with cosine schedule (α_min=0.01, α_max=0.99):
timestep:    1     16     32     64     96    128
α_t:      0.99   0.95   0.85   0.60   0.25   0.01
p_mask:   0.01   0.05   0.15   0.40   0.75   0.99

# Meaning: early timesteps (low noise) → less masking
#          later timesteps (high noise) → more masking
```

## 🛠 **Configuration**

To use consistent scheduling:

```yaml
# configs/trajectory_nelbo_gsm8k.yaml
generation:
  steps: 128                 # This becomes max_timesteps

loss:
  trajectory_nelbo_loss: true
  alpha_schedule: "cosine"   # cosine/linear/exponential
  alpha_min: 0.01          # Minimum α (maximum noise)
  alpha_max: 0.99          # Maximum α (minimum noise)

masking:
  strategy_type: "diffusion" # Will auto-sync with NELBO parameters
  p_mask_prompt: 0.3        # Still used for prompt tokens
```

## 🔍 **Verification**

The implementation includes consistency checks:

```python
# Verify schedules match
masking_params = masking_strategy.get_alpha_schedule_params()
nelbo_params = nelbo_loss.get_performance_stats()

assert masking_params['alpha_schedule'] == nelbo_params['alpha_schedule']
assert masking_params['max_timesteps'] == nelbo_params['max_timesteps']
```

This ensures that the discrete diffusion model sees **consistent noise patterns** throughout training, leading to better convergence and more meaningful NELBO bounds.