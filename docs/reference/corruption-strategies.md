# Corruption Strategies Reference

This reference documents the corruption strategies implemented in Tabular SSL for self-supervised learning on tabular data.

## Overview

Corruption strategies are the foundation of self-supervised learning for tabular data. They create pretext tasks by transforming input data in specific ways, allowing models to learn meaningful representations without labeled data.

## Available Strategies

### VIME Corruption

**Paper**: ["VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain"](https://arxiv.org/abs/2006.06775) (NeurIPS 2020)

#### Purpose
VIME creates two complementary pretext tasks:
1. **Mask Estimation**: Predict which features were corrupted
2. **Value Imputation**: Reconstruct original feature values

#### Implementation

```python
from tabular_ssl.models.components import VIMECorruption

corruption = VIMECorruption(
    corruption_rate=0.3,
    categorical_indices=[0, 1, 2],
    numerical_indices=[3, 4, 5, 6],
    categorical_vocab_sizes={0: 100, 1: 50, 2: 20},
    numerical_distributions={3: (0.5, 1.2), 4: (10.0, 5.0)}
)

# Apply corruption
corrupted_data, mask = corruption(data)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corruption_rate` | float | 0.15 | Fraction of features to corrupt |
| `categorical_indices` | List[int] | None | Indices of categorical features |
| `numerical_indices` | List[int] | None | Indices of numerical features |
| `categorical_vocab_sizes` | Dict[int, int] | None | Vocabulary sizes for categorical features |
| `numerical_distributions` | Dict[int, Tuple[float, float]] | None | (mean, std) for numerical features |

#### Outputs

- **corrupted_data**: Input data with some features corrupted
- **mask**: Binary tensor indicating corrupted positions (1=corrupted, 0=original)

#### Feature Corruption Logic

**Categorical Features**: Replace with random value from vocabulary
```python
random_categories = torch.randint(0, vocab_size, shape)
```

**Numerical Features**: Replace with random value from feature distribution
```python
random_values = torch.normal(mean, std, shape)
```

#### Configuration

```yaml
# configs/model/corruption/vime.yaml
_target_: tabular_ssl.models.components.VIMECorruption
corruption_rate: 0.3
categorical_indices: null  # Auto-detected
numerical_indices: null    # Auto-detected
```

### SCARF Corruption

**Paper**: ["SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption"](https://arxiv.org/abs/2106.15147) (arXiv 2021)

#### Purpose
SCARF optimizes representations for contrastive learning by corrupting features through replacement with values from other samples in the batch.

#### Implementation

```python
from tabular_ssl.models.components import SCARFCorruption

corruption = SCARFCorruption(
    corruption_rate=0.6,
    corruption_strategy="random_swap"  # or "marginal_sampling"
)

# Single corruption
corrupted_data = corruption(data)

# Contrastive pairs
view1, view2 = corruption.create_contrastive_pairs(data)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corruption_rate` | float | 0.6 | Fraction of features to corrupt |
| `corruption_strategy` | str | "random_swap" | "random_swap" or "marginal_sampling" |

#### Corruption Strategies

**Random Swap**: Randomly permute feature values across samples
```python
feature_values = x[:, :, feat_idx].flatten()
perm_indices = torch.randperm(len(feature_values))
shuffled_values = feature_values[perm_indices]
```

**Marginal Sampling**: Sample from marginal distribution of each feature
```python
feature_values = x[:, :, feat_idx].flatten()
sample_indices = torch.randint(0, len(feature_values), shape)
sampled_values = feature_values[sample_indices]
```

#### Configuration

```yaml
# configs/model/corruption/scarf.yaml
_target_: tabular_ssl.models.components.SCARFCorruption
corruption_rate: 0.6
corruption_strategy: "random_swap"
temperature: 0.1  # For contrastive loss
```

### ReConTab Corruption

**Purpose**: Multi-task reconstruction-based learning combining multiple corruption types with detailed tracking for reconstruction targets.

#### Implementation

```python
from tabular_ssl.models.components import ReConTabCorruption

corruption = ReConTabCorruption(
    corruption_rate=0.15,
    categorical_indices=[0, 1, 2],
    numerical_indices=[3, 4, 5, 6],
    corruption_types=["masking", "noise", "swapping"],
    masking_strategy="random",
    noise_std=0.1,
    swap_probability=0.1
)

# Apply corruption
corrupted_data, corruption_info = corruption(data)

# Get reconstruction targets
targets = corruption.reconstruction_targets(original_data, corrupted_data, corruption_info)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corruption_rate` | float | 0.15 | Base corruption rate for masking |
| `categorical_indices` | List[int] | None | Indices of categorical features |
| `numerical_indices` | List[int] | None | Indices of numerical features |
| `corruption_types` | List[str] | ["masking", "noise", "swapping"] | Types of corruption to apply |
| `masking_strategy` | str | "random" | "random", "column_wise", or "block" |
| `noise_std` | float | 0.1 | Standard deviation for Gaussian noise |
| `swap_probability` | float | 0.1 | Probability of swapping each feature |

#### Corruption Types

**Masking**: Zero out selected features
```python
mask = torch.bernoulli(torch.full(shape, corruption_rate))
x_corrupted = x * (1 - mask)
```

**Noise Injection**: Add Gaussian noise to numerical features
```python
noise = torch.randn_like(x) * noise_std
x_corrupted = x + noise
```

**Feature Swapping**: Randomly permute features across samples
```python
perm_indices = torch.randperm(batch_size)
x_corrupted[:, :, feat_idx] = x[perm_indices, :, feat_idx]
```

#### Masking Strategies

**Random**: Randomly mask individual elements
**Column-wise**: Mask entire features (columns)
**Block**: Mask contiguous temporal blocks

#### Outputs

- **corrupted_data**: Input data with applied corruptions
- **corruption_info**: Tensor indicating corruption type for each element
  - 0: Original (no corruption)
  - 1: Masked
  - 2: Noise added
  - 3: Swapped

#### Reconstruction Targets

```python
targets = {
    "masked_values": original[mask_positions],
    "mask_positions": mask_positions,
    "denoised_values": original[noise_positions],
    "noise_positions": noise_positions,
    "unswapped_values": original[swap_positions],
    "swap_positions": swap_positions
}
```

#### Configuration

```yaml
# configs/model/corruption/recontab.yaml
_target_: tabular_ssl.models.components.ReConTabCorruption
corruption_rate: 0.15
corruption_types: ["masking", "noise", "swapping"]
masking_strategy: "random"
noise_std: 0.1
swap_probability: 0.1
```

## Simple Corruption Strategies

### Random Masking

Basic random feature masking:

```python
from tabular_ssl.models.components import RandomMasking

masking = RandomMasking(corruption_rate=0.15)
masked_data = masking(data)
```

### Gaussian Noise

Add Gaussian noise to numerical features:

```python
from tabular_ssl.models.components import GaussianNoise

noise = GaussianNoise(noise_std=0.1)
noisy_data = noise(data)
```

### Swapping Corruption

Random feature swapping between samples:

```python
from tabular_ssl.models.components import SwappingCorruption

swapping = SwappingCorruption(swap_prob=0.15)
swapped_data = swapping(data)
```

## Usage in Training

### VIME Training Loop

```python
def vime_training_step(batch):
    # Apply VIME corruption
    corrupted_data, mask = vime_corruption(batch)
    
    # Forward pass
    representations = model(corrupted_data)
    
    # VIME-specific heads
    mask_pred = mask_estimation_head(representations)
    reconstructed = value_imputation_head(representations)
    
    # Compute losses
    mask_loss = F.binary_cross_entropy_with_logits(mask_pred, mask)
    recon_loss = F.mse_loss(reconstructed, batch)
    
    return mask_loss + recon_loss
```

### SCARF Training Loop

```python
def scarf_training_step(batch):
    # Create contrastive pairs
    view1, view2 = scarf_corruption.create_contrastive_pairs(batch)
    
    # Get representations
    z1 = model(view1)
    z2 = model(view2)
    
    # Contrastive loss
    loss = contrastive_loss(z1, z2, temperature=0.1)
    return loss
```

### ReConTab Training Loop

```python
def recontab_training_step(batch):
    # Apply multi-corruption
    corrupted_data, corruption_info = recontab_corruption(batch)
    
    # Forward pass
    representations = model(corrupted_data)
    
    # Get reconstruction targets
    targets = recontab_corruption.reconstruction_targets(
        batch, corrupted_data, corruption_info
    )
    
    # Multi-task reconstruction
    losses = {}
    if "masked_values" in targets:
        pred = masked_reconstruction_head(representations)
        losses["mask"] = F.mse_loss(pred[targets["mask_positions"]], 
                                   targets["masked_values"])
    
    # ... similar for denoising and unswapping
    
    return sum(losses.values())
```

## Choosing Corruption Strategies

### VIME
**Best for**: 
- Mixed categorical/numerical tabular data
- When you want explicit mask prediction capability
- Interpretable pretext tasks

**Typical corruption rate**: 0.3

### SCARF
**Best for**:
- Large datasets with diverse feature distributions
- When contrastive learning is preferred
- High-dimensional tabular data

**Typical corruption rate**: 0.6+

### ReConTab
**Best for**:
- Complex multi-task scenarios
- When you want fine-grained corruption control
- Combining reconstruction with other objectives

**Typical corruption rate**: 0.15 (base rate, actual varies by corruption type)

## Performance Tips

1. **Corruption Rate**: Start with paper defaults, then tune based on validation performance
2. **Feature Types**: Ensure correct categorical/numerical feature specification
3. **Batch Size**: SCARF benefits from larger batch sizes (128+) for effective contrastive learning
4. **Mixed Precision**: All strategies support `precision: 16-mixed` for faster training
5. **Distribution Estimation**: For VIME, set feature distributions from training data for best results

## Demo Scripts

Run interactive demos to understand each strategy:

```bash
# Compare all strategies interactively
python demo_corruption_strategies.py

# Real data demo
python demo_credit_card_data.py
```

## Paper References

- **VIME**: Yoon, J., Zhang, Y., Jordon, J., & van der Schaar, M. (2020). VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain. *NeurIPS 2020*.

- **SCARF**: Bahri, D., Jiang, H., Tay, Y., & Metzler, D. (2021). SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption for Representation Learning. *arXiv:2106.15147*. 