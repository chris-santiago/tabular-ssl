# How to Train Models Effectively

This guide solves specific problems you might encounter when training Tabular SSL models. Each section addresses a common training challenge with practical solutions.

## Problem: Your First Training Run

**Goal**: Get a model training successfully from scratch

**Solution**: Use pre-configured experiments

```bash
# Start with the simplest baseline
python train.py +experiment=simple_mlp

# Check outputs directory for results
ls outputs/$(date +%Y-%m-%d)/
```

**Why this works**: Pre-configured experiments have tested hyperparameters and compatible component combinations.

---

## Problem: Training Takes Too Long

**Goal**: Speed up training without sacrificing too much performance

### Solution 1: Reduce Model Complexity

```bash
# Use smaller model components
python train.py model/sequence_encoder=null data.batch_size=128
```

### Solution 2: Use Mixed Precision Training

```bash
# Enable 16-bit precision for faster training
python train.py +experiment=simple_mlp trainer.precision=16-mixed
```

### Solution 3: Optimize Data Loading

```bash
# Increase data workers and enable memory pinning
python train.py data.num_workers=8 data.pin_memory=true
```

---

## Problem: Running Out of GPU Memory

**Goal**: Fit your model in available GPU memory

### Solution 1: Reduce Batch Size

```bash
# Halve the batch size
python train.py +experiment=transformer_small data.batch_size=32

# Use gradient accumulation to maintain effective batch size
python train.py data.batch_size=32 trainer.accumulate_grad_batches=2
```

### Solution 2: Use Smaller Models

```bash
# Switch to more memory-efficient sequence encoder
python train.py model/sequence_encoder=rnn
```

### Solution 3: Enable Gradient Checkpointing

```yaml
# configs/experiments/memory_efficient.yaml
# @package _global_
defaults:
  - override /model/sequence_encoder: transformer

model:
  sequence_encoder:
    enable_checkpointing: true
```

---

## Problem: Model Isn't Learning (Loss Not Decreasing)

**Goal**: Debug and fix training issues

### Solution 1: Check Learning Rate

```bash
# Try different learning rates
python train.py +experiment=simple_mlp model.learning_rate=1e-2  # Higher
python train.py +experiment=simple_mlp model.learning_rate=1e-5  # Lower
```

### Solution 2: Verify Data Loading

```bash
# Use simple data config to test
python train.py +experiment=simple_mlp data=simple

# Enable debug mode to see data shapes
python train.py +experiment=simple_mlp debug=true
```

### Solution 3: Add Gradient Clipping

```bash
# Prevent exploding gradients
python train.py +experiment=simple_mlp trainer.gradient_clip_val=1.0
```

---

## Problem: Training Stops Early Due to Errors

**Goal**: Resolve common training errors

### Solution 1: Dimension Mismatches

```bash
# Check component compatibility in config files
python train.py --config-name=config +experiment=simple_mlp --print-config
```

### Solution 2: CUDA Errors

```bash
# Force CPU training for debugging
python train.py +experiment=simple_mlp trainer.accelerator=cpu

# Or specify GPU explicitly
python train.py +experiment=simple_mlp trainer.devices=1
```

---

## Problem: Need to Monitor Training Progress

**Goal**: Track training metrics and visualize progress

### Solution 1: Enable Logging

```bash
# Use Weights & Biases
python train.py +experiment=simple_mlp logger=wandb

# Use CSV logging for local development
python train.py +experiment=simple_mlp logger=csv
```

### Solution 2: Add Callbacks

```yaml
# configs/experiments/monitored_training.yaml
defaults:
  - /callbacks: default
  
callbacks:
  model_checkpoint:
    monitor: "val/loss"
    save_top_k: 3
  early_stopping:
    monitor: "val/loss"
    patience: 10
    min_delta: 0.001
```

### Solution 3: Custom Progress Logging

```bash
# Increase logging frequency
python train.py +experiment=simple_mlp trainer.log_every_n_steps=10
```

---

## Problem: Hyperparameter Tuning

**Goal**: Find optimal hyperparameters systematically

### Solution 1: Manual Grid Search

```bash
# Test different learning rates
for lr in 1e-2 1e-3 1e-4; do
    python train.py +experiment=simple_mlp model.learning_rate=$lr
done
```

### Solution 2: Use Hydra Multirun

```bash
# Test multiple hyperparameters simultaneously
python train.py -m +experiment=simple_mlp \
    model.learning_rate=1e-2,1e-3,1e-4 \
    data.batch_size=32,64,128
```

### Solution 3: Structured Experiment Sweeps

```yaml
# configs/experiments/sweep_transformer.yaml
# @package _global_
defaults:
  - override /model/sequence_encoder: transformer

# Hydra sweep configuration
hydra:
  mode: MULTIRUN
  sweeper:
    params:
      model.learning_rate: 1e-2,1e-3,1e-4
      model.sequence_encoder.num_layers: 2,4,6
      model.sequence_encoder.num_heads: 4,8
```

---

## Problem: Reproducible Training

**Goal**: Get consistent results across training runs

### Solution 1: Set Seeds Properly

```bash
# Use fixed seed
python train.py +experiment=simple_mlp seed=42

# Different experiments with different seeds
python train.py +experiment=simple_mlp seed=42,123,456
```

### Solution 2: Enable Deterministic Training

```yaml
# configs/trainer/deterministic.yaml
deterministic: true
benchmark: false
```

### Solution 3: Version Control Configurations

```bash
# Save exact configuration with each run
python train.py +experiment=simple_mlp print_config=true
```

---

## Problem: Validating Model Performance

**Goal**: Properly evaluate your trained model

### Solution 1: Enable Testing Phase

```bash
# Train and test in one command
python train.py +experiment=simple_mlp train=true test=true
```

### Solution 2: Test Only Mode

```bash
# Load and test a trained model
python train.py +experiment=simple_mlp train=false test=true \
    ckpt_path=outputs/2024-01-01/12-00-00/.../checkpoints/last.ckpt
```

### Solution 3: Cross-Validation

```bash
# Split data differently for validation
python train.py +experiment=simple_mlp \
    data.train_val_test_split=[0.6,0.2,0.2]
```

---

## Problem: Custom Training Configurations

**Goal**: Create reusable custom training setups

### Solution: Create Your Own Experiment

```yaml
# configs/experiments/my_setup.yaml
# @package _global_
defaults:
  - override /model: my_model_config
  - override /data: my_data_config

tags: ["custom", "production"]

model:
  learning_rate: 1e-4
  weight_decay: 0.01

trainer:
  max_epochs: 100
  gradient_clip_val: 0.5
  precision: 16-mixed

data:
  batch_size: 64
  num_workers: 4
```

Then use it:
```bash
python train.py +experiment=my_setup
```

---

## Quick Reference: Common Training Commands

```bash
# Fast development run
python train.py +experiment=simple_mlp trainer.max_epochs=1 data.batch_size=8

# Production training
python train.py +experiment=transformer_small logger=wandb

# Memory-efficient training
python train.py +experiment=simple_mlp trainer.precision=16-mixed data.batch_size=32

# Debug run
python train.py +experiment=simple_mlp debug=true trainer.limit_train_batches=10

# Hyperparameter sweep
python train.py -m +experiment=simple_mlp model.learning_rate=1e-2,1e-3,1e-4
```

## Next Steps

- **Evaluation**: [How to Evaluate Models](evaluation.md)
- **Data Issues**: [How to Prepare Data](data-preparation.md)
- **Advanced**: [How to Configure Experiments](configuring-experiments.md) 