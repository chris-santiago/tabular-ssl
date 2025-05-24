# Tabular SSL Model Components

This directory contains the core model components for the Tabular SSL library, including encoders, heads, and corruption strategies for self-supervised learning on tabular data.

## Architecture Overview

The Tabular SSL model follows a modular architecture:

```
Raw Tabular Data
       ↓
   Event Encoder (process individual events/transactions)
       ↓
 Sequence Encoder (model temporal dependencies)  
       ↓
  Projection Head (task-specific projection)
       ↓
  Prediction Head (final task output)
```

## Core Components

### Event Encoders

Transform individual events/transactions into dense representations:

- **`MLPEventEncoder`**: Multi-layer perceptron for mixed tabular features
- **`AutoEncoderEventEncoder`**: Autoencoder-based with reconstruction capabilities
- **`ContrastiveEventEncoder`**: Specialized for contrastive learning tasks

### Sequence Encoders

Model temporal dependencies between events in sequences:

- **`TransformerSequenceEncoder`**: Self-attention based encoder for complex dependencies
- **`RNNSequenceEncoder`**: LSTM/GRU/RNN options for sequential patterns  
- **`S4SequenceEncoder`**: Structured State Space (S4) model for long sequences

### Embedding Layers

Handle categorical variables with learned embeddings:

- **`CategoricalEmbedding`**: Flexible embedding dimensions per categorical feature

### Projection and Prediction Heads

- **`MLPProjectionHead`**: Multi-layer perceptron for representation projection
- **`ClassificationHead`**: Classification with configurable architecture

## Corruption Strategies

Implementation of state-of-the-art corruption strategies from major tabular SSL papers:

### VIME Corruption (`VIMECorruption`)

Based on ["VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain"](https://arxiv.org/abs/2006.06775)

**Key Features:**
- Binary mask generation for mask estimation pretext task
- Feature-aware corruption (categorical vs numerical)
- Value imputation from learned feature distributions
- Returns both corrupted data and corruption masks

**Usage:**
```python
from tabular_ssl.models.components import VIMECorruption

corruption = VIMECorruption(
    corruption_rate=0.3,
    categorical_indices=[0, 1, 2],
    numerical_indices=[3, 4, 5, 6]
)

# Set feature distributions from training data
corruption.set_feature_distributions(data, categorical_indices, numerical_indices)

# Apply corruption
corrupted_data, mask = corruption(data)
```

**Configuration:** `configs/model/corruption/vime.yaml`

### SCARF Corruption (`SCARFCorruption`)

Based on ["SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption"](https://arxiv.org/abs/2106.15147)

**Key Features:**
- Random feature corruption by replacing with values from other samples
- Optimized for contrastive learning
- High corruption rates (typically 60%+)
- Creates multiple corrupted views for contrastive pairs

**Usage:**
```python
from tabular_ssl.models.components import SCARFCorruption

corruption = SCARFCorruption(
    corruption_rate=0.6,
    corruption_strategy="random_swap"  # or "marginal_sampling"
)

# Apply corruption  
corrupted_data = corruption(data)

# Create contrastive pairs
view1, view2 = corruption.create_contrastive_pairs(data)
```

**Configuration:** `configs/model/corruption/scarf.yaml`

### ReConTab Corruption (`ReConTabCorruption`)

Based on reconstruction-based contrastive learning approaches for tabular data.

**Key Features:**
- Multiple corruption types: masking, noise injection, feature swapping
- Detailed corruption tracking for reconstruction targets
- Flexible masking strategies: random, column-wise, block-wise
- Separate treatment of categorical and numerical features

**Usage:**
```python
from tabular_ssl.models.components import ReConTabCorruption

corruption = ReConTabCorruption(
    corruption_rate=0.15,
    corruption_types=["masking", "noise", "swapping"],
    masking_strategy="random",
    noise_std=0.1
)

# Apply corruption
corrupted_data, corruption_info = corruption(data)

# Get reconstruction targets
targets = corruption.reconstruction_targets(original_data, corrupted_data, corruption_info)
```

**Configuration:** `configs/model/corruption/recontab.yaml`

## Basic Corruption Strategies

Additional simple corruption strategies:

- **`RandomMasking`**: Basic random feature masking
- **`GaussianNoise`**: Gaussian noise injection for numerical features
- **`SwappingCorruption`**: Random feature swapping between samples

## Experiments

Pre-configured experiments demonstrate each corruption strategy:

### VIME SSL Experiment
```bash
python train.py +experiment=vime_ssl
```

**Features:**
- VIME corruption with mask estimation and value imputation tasks
- Transformer sequence encoder for temporal patterns
- Multi-task learning with weighted losses

### SCARF SSL Experiment  
```bash
python train.py +experiment=scarf_ssl
```

**Features:**
- SCARF corruption optimized for contrastive learning
- Large batch sizes for effective contrastive training
- InfoNCE loss with temperature scaling

### ReConTab SSL Experiment
```bash
python train.py +experiment=recontab_ssl
```

**Features:**
- Multi-type corruption with reconstruction targets
- Combination of reconstruction and contrastive losses
- Flexible corruption scheduling

## Demo Scripts

### Corruption Strategies Demo
```bash
python demo_corruption_strategies.py
```

Interactive demonstration showing:
- How each corruption strategy works
- Input/output formats and shapes
- Corruption rate analysis
- Side-by-side comparison

### Credit Card Data Demo
```bash
python demo_credit_card_data.py
```

End-to-end demonstration with real transaction data:
- Downloads IBM TabFormer credit card dataset
- Tests DataModule integration
- Shows training readiness

## Usage Examples

### Basic Model Assembly

```python
from tabular_ssl.models.components import (
    MLPEventEncoder, TransformerSequenceEncoder, 
    MLPProjectionHead, ClassificationHead
)

# Create model components
event_encoder = MLPEventEncoder(
    input_dim=64,
    hidden_dims=[128, 256],
    output_dim=512
)

sequence_encoder = TransformerSequenceEncoder(
    input_dim=512,
    hidden_dim=512,
    num_layers=4,
    num_heads=8
)

projection_head = MLPProjectionHead(
    input_dim=512,
    hidden_dims=[256, 128],
    output_dim=64
)

# Process data
x = event_encoder(tabular_data)
x = sequence_encoder(x) 
representations = projection_head(x)
```

### Self-Supervised Training Loop

```python
from tabular_ssl.models.components import VIMECorruption

# Initialize corruption
corruption = VIMECorruption(corruption_rate=0.3)

# Training step
def train_step(batch):
    # Apply corruption
    corrupted_data, mask = corruption(batch)
    
    # Forward pass
    representations = model(corrupted_data)
    
    # VIME losses
    mask_pred = mask_estimation_head(representations)
    reconstructed = value_imputation_head(representations)
    
    mask_loss = F.binary_cross_entropy_with_logits(mask_pred, mask)
    recon_loss = F.mse_loss(reconstructed, batch)
    
    total_loss = mask_loss + recon_loss
    return total_loss
```

## Configuration

All components support Hydra configuration:

```yaml
# Model configuration
model:
  event_encoder:
    _target_: tabular_ssl.models.components.MLPEventEncoder
    input_dim: 64
    hidden_dims: [128, 256, 512] 
    output_dim: 512
    
  sequence_encoder:
    _target_: tabular_ssl.models.components.TransformerSequenceEncoder
    input_dim: 512
    hidden_dim: 512
    num_layers: 4
    
  corruption:
    _target_: tabular_ssl.models.components.VIMECorruption
    corruption_rate: 0.3
```

## Advanced Features

### Categorical Embedding Management

```python
from tabular_ssl.models.components import CategoricalEmbedding

# Flexible embedding dimensions per feature
embedding = CategoricalEmbedding(
    vocab_sizes={"merchant": 1000, "category": 50},
    embedding_dims={"merchant": 64, "category": 16}
)
```

### S4 for Long Sequences

```python
from tabular_ssl.models.components import S4SequenceEncoder

# Efficient processing of very long sequences
s4_encoder = S4SequenceEncoder(
    input_dim=256,
    hidden_dim=256,
    num_layers=6,
    bidirectional=True,
    max_sequence_length=4096
)
```

### Multi-Task Heads

```python
# Different prediction heads for different tasks
fraud_head = ClassificationHead(input_dim=128, num_classes=2)
amount_head = MLPProjectionHead(input_dim=128, output_dim=1)
category_head = ClassificationHead(input_dim=128, num_classes=20)
```

## Performance Tips

1. **Batch Size**: Use larger batches (128+) for contrastive methods like SCARF
2. **Corruption Rate**: Start with paper defaults (VIME: 0.3, SCARF: 0.6, ReConTab: 0.15)
3. **Sequence Length**: Longer sequences benefit from S4 encoder
4. **Mixed Precision**: Use `precision: 16-mixed` for faster training
5. **Gradient Clipping**: Set `gradient_clip_val: 1.0` for stability

## Paper References

- **VIME**: Yoon, J., Zhang, Y., Jordon, J., & van der Schaar, M. (2020). VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain. *NeurIPS 2020*.

- **SCARF**: Bahri, D., Jiang, H., Tay, Y., & Metzler, D. (2021). SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption for Representation Learning. *arXiv:2106.15147*.

- **S4**: Gu, A., Goel, K., & Ré, C. (2021). Efficiently Modeling Long Sequences with Structured State Spaces. *ICLR 2022*.

## Contributing

When adding new components:

1. **Inherit from base classes** in `base.py`
2. **Add comprehensive docstrings** with parameter descriptions
3. **Create configuration files** in `configs/model/`
4. **Add unit tests** for new functionality
5. **Update this README** with usage examples

For corruption strategies:
1. **Follow paper implementations** closely
2. **Support both categorical and numerical features**
3. **Return appropriate metadata** (masks, corruption info, etc.)
4. **Add demo examples** showing the corruption behavior 