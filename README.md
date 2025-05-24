# Tabular SSL: Self-Supervised Learning for Tabular Data

A modular framework for self-supervised learning on tabular data with **state-of-the-art corruption strategies** and **ready-to-use sample data**.

## 🎭 New: State-of-the-Art Corruption Strategies

We've implemented corruption strategies from leading tabular SSL papers:

- **🎯 VIME** - Value imputation and mask estimation ([NeurIPS 2020](https://arxiv.org/abs/2006.06775))
- **🌟 SCARF** - Contrastive learning with feature corruption ([arXiv 2021](https://arxiv.org/abs/2106.15147))
- **🔧 ReConTab** - Multi-task reconstruction-based learning

## 🚀 Quick Start

Try our interactive demos to see the corruption strategies in action:

```bash
# Demo corruption strategies (VIME, SCARF, ReConTab)
python demo_corruption_strategies.py

# Demo with real credit card transaction data
python demo_credit_card_data.py

# Train with state-of-the-art SSL methods
python train.py +experiment=vime_ssl     # VIME approach
python train.py +experiment=scarf_ssl    # SCARF approach  
python train.py +experiment=recontab_ssl # ReConTab approach
```

## Overview

Tabular SSL provides a flexible framework for self-supervised learning on tabular data, with support for:
- **🎭 State-of-the-art corruption strategies** (VIME, SCARF, ReConTab)
- **🏦 Ready-to-use sample data** (IBM TabFormer credit card transactions)
- **📱 Interactive demos** to understand corruption strategies
- Event sequence modeling with multiple encoder types (RNN, LSTM, GRU, Transformer, S4)
- Modular architecture with Hydra configuration
- PyTorch Lightning for robust training

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tabular-ssl.git
cd tabular-ssl

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Project Structure

```
tabular-ssl/
├── configs/                        # Hydra configuration files
│   ├── config.yaml                # Main configuration
│   ├── model/                     # Model configurations
│   │   ├── event_encoder/        # Event encoder configs
│   │   ├── sequence_encoder/     # Sequence encoder configs
│   │   ├── embedding/            # Embedding configs
│   │   ├── projection_head/      # Projection head configs
│   │   └── prediction_head/      # Prediction head configs
│   ├── data/                     # Data configurations
│   ├── trainer/                  # Training configurations
│   ├── callbacks/                # Callback configurations
│   ├── logger/                   # Logger configurations
│   ├── experiment/               # Experiment configurations
│   ├── hydra/                    # Hydra-specific configurations
│   └── paths/                    # Path configurations
├── src/
│   ├── tabular_ssl/              # Core package
│   │   ├── data/                # Data loading and processing
│   │   ├── models/              # Model implementations
│   │   │   ├── base.py         # Base model and component registry
│   │   │   ├── components.py   # Model components
│   │   │   └── s4.py           # S4 implementation
│   │   └── utils/              # Utility functions
│   └── train.py                  # Training script
└── tests/                         # Unit tests
```

## Component Registry

The project uses a component registry pattern that allows for:

1. **Modular Components**: Easy to add, modify, or extend components
2. **Configuration Validation**: Type-safe configuration with Pydantic
3. **Dynamic Component Loading**: Components are loaded at runtime based on configuration

### How the Registry Works

All model components are registered with the `ComponentRegistry`:

```python
@ComponentRegistry.register("mlp_event_encoder")
class MLPEventEncoder(EventEncoder):
    def __init__(self, config: MLPConfig):
        super().__init__(config)
        # Implementation...
```

Components can then be instantiated from configuration:

```python
# Get component class
component_cls = ComponentRegistry.get("mlp_event_encoder")

# Create configuration
config = MLPConfig(
    name="mlp_encoder",
    type="mlp_event_encoder",
    input_dim=64,
    hidden_dims=[128, 256],
    output_dim=512
)

# Instantiate component
component = component_cls(config)
```

### Available Components

#### Event Encoders
- `mlp_event_encoder`: MLP-based event encoder
- `autoencoder`: Autoencoder-based event encoder
- `contrastive`: Contrastive learning event encoder

#### Sequence Encoders
- `rnn`: Basic RNN encoder
- `lstm`: LSTM encoder
- `gru`: GRU encoder
- `transformer`: Transformer encoder
- `s4`: Diagonal State Space Model (S4) encoder

#### Embedding Layers
- `categorical_embedding`: Embedding layer for categorical variables

#### Projection Heads
- `mlp_projection`: MLP-based projection head

#### Prediction Heads
- `classification`: Classification head

#### Corruption Strategies
- `random_masking`: Random masking corruption
- `gaussian_noise`: Gaussian noise corruption
- `swapping`: Feature swapping corruption
- `vime`: VIME-style corruption
- `corruption_pipeline`: Pipeline of multiple corruption strategies

## Configuration with Hydra

The project uses [Hydra](https://hydra.cc/) for configuration management, allowing for:

1. **Hierarchical Configuration**: Configurations are organized into groups
2. **Command-line Overrides**: Parameters can be changed via command line
3. **Configuration Composition**: Mix and match configurations for experiments
4. **Multirun**: Run parameter sweeps and experiments

### Basic Usage

```bash
# Train with default configuration
python src/train.py

# Override specific parameters
python src/train.py model.optimizer.lr=0.001 data.batch_size=32

# Use a specific experiment configuration
python src/train.py experiment=s4_sequence

# Run in debug mode
python src/train.py debug=true
```

### Example: Creating a Custom Component

1. **Create a new component class**:

```python
# src/tabular_ssl/models/components.py
from .base import ComponentRegistry, EventEncoder, ComponentConfig
from pydantic import Field

class CustomEncoderConfig(ComponentConfig):
    input_dim: int = Field(..., description="Input dimension")
    output_dim: int = Field(..., description="Output dimension")
    # Add custom parameters...

@ComponentRegistry.register("custom_encoder")
class CustomEncoder(EventEncoder):
    def __init__(self, config: CustomEncoderConfig):
        super().__init__(config)
        # Implementation...
        
    def forward(self, x):
        # Implementation...
        return encoded
```

2. **Create a configuration file**:

```yaml
# configs/model/event_encoder/custom.yaml
name: custom_encoder
type: custom_encoder
input_dim: 64
output_dim: 32
# Add custom parameters...
```

3. **Use in experiments**:

```bash
python src/train.py model/event_encoder=custom
```

### Example: Creating an Experiment

Create a new experiment configuration:

```yaml
# configs/experiment/custom_experiment.yaml
# @package _global_

# to execute this experiment run:
# python train.py experiment=custom_experiment

defaults:
  - override /model/event_encoder: custom.yaml
  - override /model/sequence_encoder: s4.yaml
  - override /trainer: default.yaml
  - override /model: default.yaml
  - _self_

# all parameters below will be merged with parameters from default configurations set above
# this allows you to overwrite only specific parameters

tags: ["custom", "s4"]

seed: 12345

trainer:
  max_epochs: 50
  gradient_clip_val: 0.5

model:
  optimizer:
    lr: 1.0e-4
    weight_decay: 0.01
```

## Creating Self-Supervised Learning Tasks

The framework supports various self-supervised learning tasks:

### 1. Reconstruction-based (Autoencoder)

```yaml
# configs/model/event_encoder/autoencoder.yaml
name: autoencoder_encoder
type: autoencoder
input_dim: 64
hidden_dims: [128, 64]
output_dim: 32
dropout: 0.1
use_batch_norm: true
use_reconstruction_loss: true
```

### 2. Contrastive Learning

```yaml
# configs/model/event_encoder/contrastive.yaml
name: contrastive_encoder
type: contrastive
input_dim: 64
hidden_dims: [128, 64]
output_dim: 32
dropout: 0.1
use_batch_norm: true
temperature: 0.07
```

### 3. Feature Corruption (VIME-style)

```yaml
# Create a corruption pipeline in your model's training_step
corruption_config = CorruptionPipelineConfig(
    name="corruption_pipeline",
    type="corruption_pipeline",
    strategies=["random_masking", "gaussian_noise"],
    corruption_rates=[0.15, 0.1]
)
corruption = ComponentRegistry.get("corruption_pipeline")(corruption_config)

# Apply corruption
x_corrupted = corruption(x)
```

## Extending the Framework

### Adding New Components

1. Create a new configuration class that inherits from `ComponentConfig`
2. Create a new component class that inherits from the appropriate base class
3. Register the component with `@ComponentRegistry.register("component_name")`
4. Create a configuration file in the appropriate directory

### Adding New Experiments

1. Create a new experiment configuration file in `configs/experiment/`
2. Use the `defaults` section to override component configurations
3. Add experiment-specific parameters

### Adding New Metrics

Add custom metrics in your model's training and validation steps:

```python
def training_step(self, batch, batch_idx):
    # Your training logic
    loss = ...
    
    # Log metrics
    self.log("train/loss", loss)
    self.log("train/accuracy", accuracy)
    
    return loss
```

## References

- [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)
- [Hydra](https://hydra.cc/)
- [S4: Structured State Space Sequence Models](https://arxiv.org/abs/2111.00396)
- [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template) 