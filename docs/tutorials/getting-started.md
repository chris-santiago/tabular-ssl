# Getting Started with Tabular SSL

This tutorial will guide you through the process of setting up and using the Tabular SSL library for self-supervised learning on tabular data.

## Installation

First, clone and install the library:

```bash
# Clone the repository
git clone https://github.com/yourusername/tabular-ssl.git
cd tabular-ssl

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Set PYTHONPATH for imports
export PYTHONPATH=$PWD/src
```

## Basic Usage

Here's a simple example using pre-configured experiments:

```bash
# Run MLP-only baseline
python train.py +experiment=simple_mlp

# Run transformer experiment
python train.py +experiment=transformer_small

# Run with different data configuration
python train.py +experiment=simple_mlp data=simple

# Run with logging
python train.py +experiment=simple_mlp logger=wandb
```

## Step-by-Step Guide

### 1. Understanding the Configuration System

Tabular SSL uses Hydra for configuration management. The main config structure is:

```yaml
# configs/config.yaml
defaults:
  - data: default
  - model: default  
  - trainer: default
  - logger: null

# Training settings
train: true
test: false
seed: 42
```

### 2. Data Configuration

Configure your data module:

```yaml
# configs/data/default.yaml
_target_: tabular_ssl.data.datamodule.TabularDataModule

data_dir: ${paths.data_dir}
dataset_name: sample_dataset
batch_size: 64
sequence_length: 32

feature_config:
  categorical_features:
    - name: category_1
      num_categories: 10
  numerical_features:
    - name: value_1
      mean: 0.0
      std: 1.0
```

### 3. Model Configuration

The model is composed of modular components:

```yaml
# configs/model/default.yaml
defaults:
  - event_encoder: mlp
  - sequence_encoder: transformer
  - projection_head: mlp
  - prediction_head: classification

_target_: tabular_ssl.models.base.BaseModel

learning_rate: 1.0e-4
weight_decay: 0.01
optimizer_type: adamw
```

### 4. Running Experiments

Use pre-configured experiments:

```bash
# Simple MLP baseline
python train.py +experiment=simple_mlp

# Transformer for sequence modeling
python train.py +experiment=transformer_small

# S4 for long sequences
python train.py +experiment=s4_large

# RNN baseline
python train.py +experiment=rnn_baseline
```

### 5. Custom Configurations

Override specific components:

```bash
# Use RNN instead of transformer
python train.py model/sequence_encoder=rnn

# No sequence encoder (MLP only)
python train.py model/sequence_encoder=null

# Custom learning rate
python train.py model.learning_rate=1e-3

# Different batch size
python train.py data.batch_size=128
```

### 6. Creating Custom Training Script

```python
# train.py
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: DictConfig):
    # Set seed
    pl.seed_everything(config.seed)
    
    # Create data module
    datamodule = hydra.utils.instantiate(config.data)
    
    # Create model
    model = hydra.utils.instantiate(config.model)
    
    # Create trainer
    trainer = hydra.utils.instantiate(config.trainer)
    
    # Train
    if config.train:
        trainer.fit(model, datamodule=datamodule)
    
    # Test
    if config.test:
        trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
```

## Available Components

### Event Encoders
- **MLP**: Multi-layer perceptron with configurable architecture

### Sequence Encoders
- **Transformer**: Self-attention based sequence modeling
- **S4**: Structured state space model for long sequences
- **RNN**: LSTM/GRU for sequential processing
- **null**: No sequence processing (MLP-only)

### Other Components
- **Embedding**: Categorical feature embeddings with flexible dimensions
- **Projection Head**: MLP for representation projection
- **Prediction Head**: Classification/regression head

## Next Steps

- Check out the [Basic Usage](basic-usage.md) tutorial for more advanced examples
- Explore the [API Reference](../reference/api.md) for detailed documentation
- Learn about different [SSL Methods](../explanation/ssl-methods.md) you can use

## Common Issues and Solutions

### Import Errors

Make sure PYTHONPATH is set correctly:

```bash
export PYTHONPATH=/path/to/tabular-ssl/src
```

### Configuration Errors

Validate your configuration:

```bash
# Print configuration without running
python train.py --print-config

# Test specific experiment
python train.py +experiment=simple_mlp --print-config
```

### Memory Issues

Use smaller models or batch sizes:

```bash
# Smaller batch size
python train.py data.batch_size=32

# Simple MLP (no sequence processing)
python train.py model/sequence_encoder=null

# Use simple data configuration
python train.py data=simple
```

## Additional Resources

- [GitHub Repository](https://github.com/yourusername/tabular-ssl)
- [Issue Tracker](https://github.com/yourusername/tabular-ssl/issues)
- [Contributing Guide](../CONTRIBUTING.md) 