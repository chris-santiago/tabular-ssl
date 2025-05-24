# How-to: Self-Supervised Learning Training

This guide covers advanced techniques for training self-supervised learning models with Tabular SSL's corruption strategies.

## Quick Start

### Run Demo Scripts First

Before training, explore the corruption strategies interactively:

```bash
# See how corruption strategies work
python demo_corruption_strategies.py

# Try with real credit card data
python demo_credit_card_data.py
```

### Basic SSL Training

```bash
# VIME: Mask estimation + value imputation
python train.py +experiment=vime_ssl

# SCARF: Contrastive learning
python train.py +experiment=scarf_ssl

# ReConTab: Multi-task reconstruction
python train.py +experiment=recontab_ssl
```

## Choosing the Right Strategy

### VIME - When to Use

**Best for:**
- Mixed categorical/numerical tabular data
- Interpretable pretext tasks
- Downstream tasks requiring feature reconstruction

**Characteristics:**
- Moderate corruption rate (30%)
- Returns explicit masks
- Two complementary tasks: mask estimation + value imputation

```bash
python train.py +experiment=vime_ssl
```

### SCARF - When to Use

**Best for:**
- Large datasets with diverse features
- Pure representation learning
- High-dimensional tabular data

**Characteristics:**
- High corruption rate (60%+)
- Contrastive learning approach
- Requires larger batch sizes

```bash
python train.py +experiment=scarf_ssl
```

### ReConTab - When to Use

**Best for:**
- Complex multi-task scenarios
- Fine-grained corruption control
- Hybrid reconstruction + contrastive approaches

**Characteristics:**
- Low base corruption rate (15%) but multiple types
- Detailed corruption tracking
- Flexible masking strategies

```bash
python train.py +experiment=recontab_ssl
```

## Customizing Corruption Parameters

### VIME Customization

```bash
# Adjust corruption rate
python train.py +experiment=vime_ssl model/corruption.corruption_rate=0.5

# Use with different sequence lengths
python train.py +experiment=vime_ssl data.sequence_length=64

# Modify loss weights
python train.py +experiment=vime_ssl model.mask_estimation_weight=2.0 model.value_imputation_weight=1.0
```

### SCARF Customization

```bash
# Change corruption strategy
python train.py +experiment=scarf_ssl model/corruption.corruption_strategy=marginal_sampling

# Adjust temperature for contrastive loss
python train.py +experiment=scarf_ssl model.temperature=0.05

# Use larger batch size (important for SCARF)
python train.py +experiment=scarf_ssl data.batch_size=256
```

### ReConTab Customization

```bash
# Enable only specific corruption types
python train.py +experiment=recontab_ssl model/corruption.corruption_types=['masking','noise']

# Use column-wise masking
python train.py +experiment=recontab_ssl model/corruption.masking_strategy=column_wise

# Adjust individual corruption parameters
python train.py +experiment=recontab_ssl model/corruption.noise_std=0.2 model/corruption.swap_probability=0.2
```

## Working with Your Own Data

### Preparing Data for SSL

1. **Create your DataModule:**

```python
# configs/data/your_data.yaml
_target_: tabular_ssl.data.base.DataModule
data_path: "path/to/your/data.csv"
sequence_length: 32
batch_size: 64

# Feature specifications
categorical_columns: ["category_col1", "category_col2"]
numerical_columns: ["num_col1", "num_col2", "num_col3"]

# Sample data generation (optional)
sample_data_config:
  n_users: 1000
  sequence_length: 32
```

2. **Use with SSL experiments:**

```bash
# Use your data with VIME
python train.py +experiment=vime_ssl data=your_data

# Use your data with SCARF
python train.py +experiment=scarf_ssl data=your_data
```

### Feature Type Detection

Corruption strategies need to know which features are categorical vs numerical:

```python
# Automatic detection (default)
python train.py +experiment=vime_ssl

# Manual specification
python train.py +experiment=vime_ssl \
  model/corruption.categorical_indices=[0,1,2] \
  model/corruption.numerical_indices=[3,4,5,6,7]
```

## Advanced Training Techniques

### Multi-GPU Training

```bash
# Use multiple GPUs
python train.py +experiment=vime_ssl trainer.devices=2 trainer.strategy=ddp

# Adjust batch size for multi-GPU
python train.py +experiment=scarf_ssl trainer.devices=4 data.batch_size=512
```

### Mixed Precision Training

All SSL experiments support mixed precision for faster training:

```bash
# Already enabled in experiments (precision: 16-mixed)
python train.py +experiment=vime_ssl

# Disable if needed
python train.py +experiment=vime_ssl trainer.precision=32
```

### Hyperparameter Optimization

Use Hydra's multirun for hyperparameter sweeps:

```bash
# Sweep corruption rates for VIME
python train.py +experiment=vime_ssl -m model/corruption.corruption_rate=0.1,0.3,0.5

# Sweep SCARF parameters
python train.py +experiment=scarf_ssl -m \
  model/corruption.corruption_rate=0.4,0.6,0.8 \
  model.temperature=0.05,0.1,0.2
```

## Monitoring Training

### Key Metrics to Watch

**VIME:**
- `train/mask_estimation_loss` - Should decrease steadily
- `train/value_imputation_loss` - Should decrease steadily
- `val/total_loss` - Overall validation performance

**SCARF:**
- `train/contrastive_loss` - Should decrease and stabilize
- Representation quality metrics (if using downstream tasks)

**ReConTab:**
- `train/masked_reconstruction` - Masking reconstruction quality
- `train/denoising` - Noise removal quality
- `train/unswapping` - Feature unswapping quality

### Using Weights & Biases

SSL experiments are pre-configured for W&B logging:

```bash
# Logs automatically to your W&B account
python train.py +experiment=vime_ssl

# Customize project name
python train.py +experiment=vime_ssl logger.wandb.project=my-ssl-project
```

## Troubleshooting

### Poor Convergence

**Problem:** Training loss not decreasing

**Solutions:**
```bash
# Lower learning rate
python train.py +experiment=vime_ssl model.learning_rate=5e-5

# Increase warmup steps
python train.py +experiment=vime_ssl model.scheduler_type=cosine_with_warmup

# Reduce corruption rate
python train.py +experiment=vime_ssl model/corruption.corruption_rate=0.2
```

### Memory Issues

**Problem:** CUDA out of memory

**Solutions:**
```bash
# Reduce batch size
python train.py +experiment=scarf_ssl data.batch_size=32

# Reduce sequence length
python train.py +experiment=vime_ssl data.sequence_length=16

# Use gradient accumulation
python train.py +experiment=vime_ssl trainer.accumulate_grad_batches=2
```

### SCARF-Specific Issues

**Problem:** Contrastive loss not decreasing

**Solutions:**
```bash
# Increase batch size (critical for SCARF)
python train.py +experiment=scarf_ssl data.batch_size=256

# Adjust temperature
python train.py +experiment=scarf_ssl model.temperature=0.07

# Increase corruption rate
python train.py +experiment=scarf_ssl model/corruption.corruption_rate=0.8
```

## Evaluation and Downstream Tasks

### Save Trained Models

SSL experiments automatically save checkpoints:

```bash
# Training saves to outputs/YYYY-MM-DD/HH-MM-SS/
ls outputs/  # Find your experiment

# Best checkpoint is saved automatically
ls outputs/2024-01-15/14-30-45/checkpoints/
```

### Extract Representations

```python
import torch
from tabular_ssl.models.base import BaseModel

# Load trained model
model = BaseModel.load_from_checkpoint("path/to/checkpoint.ckpt")
model.eval()

# Extract representations
with torch.no_grad():
    representations = model(your_data)
```

### Downstream Task Training

Use pre-trained SSL models for downstream tasks:

```bash
# Load SSL checkpoint for fine-tuning
python train.py +experiment=classification_finetune \
  model.ssl_checkpoint_path=outputs/2024-01-15/14-30-45/checkpoints/best.ckpt
```

## Custom Corruption Strategies

### Create Your Own Strategy

```python
# custom_corruption.py
import torch
import torch.nn as nn

class CustomCorruption(nn.Module):
    def __init__(self, corruption_rate: float = 0.2):
        super().__init__()
        self.corruption_rate = corruption_rate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        
        # Your custom corruption logic
        mask = torch.rand_like(x) > self.corruption_rate
        return x * mask
```

### Use Custom Strategy

```yaml
# configs/model/corruption/custom.yaml
_target_: path.to.custom_corruption.CustomCorruption
corruption_rate: 0.2
```

```bash
python train.py model/corruption=custom
```

## Best Practices

### 1. Start with Demos
Always run `demo_corruption_strategies.py` first to understand how each strategy works.

### 2. Use Appropriate Batch Sizes
- **VIME/ReConTab**: 32-128 typically sufficient
- **SCARF**: 128+ recommended for effective contrastive learning

### 3. Monitor Feature Types
Ensure categorical/numerical indices are correctly specified for optimal corruption.

### 4. Experiment with Corruption Rates
- Start with paper defaults
- Tune based on downstream task performance
- Higher rates aren't always better

### 5. Use Mixed Precision
Enable `precision: 16-mixed` for 2x speedup with minimal quality loss.

### 6. Save Everything
SSL training can be expensive - ensure checkpointing is enabled.

## Paper References

For implementation details and theoretical background:

- **VIME**: [NeurIPS 2020](https://arxiv.org/abs/2006.06775)
- **SCARF**: [arXiv 2021](https://arxiv.org/abs/2106.15147)
- **General SSL**: [Self-Supervised Learning Survey](https://arxiv.org/abs/1902.06162) 