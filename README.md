# Tabular SSL: Self-Supervised Learning for Tabular Data

This project was created through an AI chat conversation with Cursor, demonstrating the capabilities of AI-assisted development. The project implements a modular framework for self-supervised learning on tabular data, with a focus on sequence modeling.

## Overview

Tabular SSL provides a flexible framework for self-supervised learning on tabular data, with support for:
- Event sequence modeling
- Multiple sequence encoder types (RNN, LSTM, GRU, Transformer, SSM, S4)
- Customizable feature processing pipelines
- PyTorch Lightning integration
- Hydra configuration management

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
tabular_ssl/
├── configs/                 # Hydra configuration files
│   ├── config.yaml         # Main configuration
│   ├── model/             # Model configurations
│   ├── data/              # Data configurations
│   └── trainer/           # Trainer configurations
├── tabular_ssl/
│   ├── data/              # Data loading and processing
│   ├── models/            # Model implementations
│   └── utils/             # Utility functions
└── src/                   # Source code
    └── train.py           # Training script
```

## Quick Experiments

The framework is designed for easy experimentation with different architectures and components. Here's how to run quick experiments:

### 1. Swapping Components via Command Line

```bash
# Try different sequence encoders
python src/train.py model.sequence_encoder.model_type=transformer
python src/train.py model.sequence_encoder.model_type=s4
python src/train.py model.sequence_encoder.model_type=lstm

# Modify encoder parameters
python src/train.py model.sequence_encoder.num_layers=4 model.sequence_encoder.hidden_dim=64

# Change event encoder architecture
python src/train.py model.event_encoder.hidden_dims=[128,64,32]

# Adjust training parameters
python src/train.py trainer.max_epochs=50 trainer.accelerator=gpu trainer.devices=2
```

### 2. Creating Experiment Configurations

Create new configuration files in the `configs/experiments` directory:

```yaml
# configs/experiments/s4_large.yaml
model:
  sequence_encoder:
    model_type: "s4"
    hidden_dim: 128
    num_layers: 4
    state_dim: 64
    max_sequence_length: 2048

  event_encoder:
    hidden_dims: [256, 128, 64]
    dropout: 0.2

trainer:
  max_epochs: 100
  precision: "16-mixed"
```

Run the experiment:
```bash
python src/train.py experiment=s4_large
```

### 3. Component Architecture Options

#### Event Encoders
- MLP (default)
- CNN
- Custom architectures

```yaml
model:
  event_encoder:
    type: "mlp"  # or "cnn"
    input_dim: 10
    hidden_dims: [64, 32]
    output_dim: 16
```

#### Sequence Encoders
- RNN
- LSTM
- GRU
- Transformer
- SSM (State Space Model)
- S4 (Structured State Space Sequence)

```yaml
model:
  sequence_encoder:
    model_type: "transformer"  # or "rnn", "lstm", "gru", "ssm", "s4"
    input_dim: 16
    hidden_dim: 32
    num_layers: 2
    # Model-specific parameters
    num_heads: 4  # for transformer
    state_dim: 32  # for ssm/s4
```

#### Feature Processors
- Tabular (default)
- Custom processors

```yaml
data:
  feature_config:
    processors:
      - name: "tabular"
        params:
          normalize_numerical: true
      - name: "custom"
        params:
          param1: "value1"
```

### 4. Running Experiment Sweeps

Use Hydra's multirun feature for parameter sweeps:

```bash
# Sweep over different sequence encoders
python src/train.py -m model.sequence_encoder.model_type=transformer,lstm,s4

# Sweep over multiple parameters
python src/train.py -m \
  model.sequence_encoder.model_type=transformer,s4 \
  model.sequence_encoder.hidden_dim=32,64,128 \
  trainer.max_epochs=50,100
```

### 5. Experiment Tracking

The framework integrates with various experiment trackers:

```yaml
# configs/logger/default.yaml
default:
  - _self_
  - wandb: true  # or mlflow, tensorboard
  - wandb:
      project: "tabular-ssl"
      name: ${now:%Y-%m-%d_%H-%M-%S}
      tags: ${model.sequence_encoder.model_type}
```

## Components

### 1. Event Encoders

The framework provides several event encoder options:

#### MLP Event Encoder
```python
from tabular_ssl.models import MLPEventEncoder

encoder = MLPEventEncoder({
    "input_dim": 10,
    "hidden_dims": [64, 32],
    "output_dim": 16
})
```

#### AutoEncoder Event Encoder
A reconstruction-based encoder that learns through self-supervised reconstruction:
```python
from tabular_ssl.models import AutoEncoderEventEncoder

encoder = AutoEncoderEventEncoder({
    "input_dim": 10,
    "hidden_dims": [64, 32],
    "output_dim": 16,
    "dropout": 0.1
})

# Get encoded representation
z = encoder(x)
# Get reconstruction
x_recon = encoder.decode(z)
# Get reconstruction loss
loss = encoder.reconstruction_loss(x)
```

Configuration:
```yaml
model:
  event_encoder:
    _target_: tabular_ssl.models.components.AutoEncoderEventEncoder
    input_dim: 10
    hidden_dims: [64, 32]
    output_dim: 16
    dropout: 0.1
```

#### Contrastive Event Encoder
A contrastive learning encoder that learns through comparing different views of the same input:
```python
from tabular_ssl.models import ContrastiveEventEncoder

encoder = ContrastiveEventEncoder({
    "input_dim": 10,
    "hidden_dims": [64, 32],
    "output_dim": 16,
    "dropout": 0.1,
    "temperature": 0.07
})

# Get contrastive representations
z1 = encoder(x1)  # First view
z2 = encoder(x2)  # Second view
# Get contrastive loss
loss = encoder.contrastive_loss(z1, z2)
```

Configuration:
```yaml
model:
  event_encoder:
    _target_: tabular_ssl.models.components.ContrastiveEventEncoder
    input_dim: 10
    hidden_dims: [64, 32]
    output_dim: 16
    dropout: 0.1
    temperature: 0.07
```

For contrastive learning, you'll need to create two views of your input data. Common augmentation strategies include:
- Random masking of features
- Adding Gaussian noise
- Feature dropout
- Time-based augmentations for sequential data

Example augmentation:
```python
def create_views(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # First view: Add Gaussian noise
    x1 = x + torch.randn_like(x) * 0.1
    
    # Second view: Random feature masking
    mask = torch.rand_like(x) > 0.2
    x2 = x * mask
    
    return x1, x2

# Usage in training
x1, x2 = create_views(x)
z1 = encoder(x1)
z2 = encoder(x2)
loss = encoder.contrastive_loss(z1, z2)
```

### 2. Feature Processing

The framework provides a flexible feature processing system:

```python
from tabular_ssl.data.datamodule import FeatureConfig, ProcessorConfig

# Basic configuration
feature_config = FeatureConfig(
    categorical_cols=["user_id", "item_id"],
    numerical_cols=["price", "quantity"],
    target_col="label"
)

# Advanced configuration with multiple processors
feature_config = FeatureConfig(
    categorical_cols=["user_id", "item_id"],
    numerical_cols=["price", "quantity"],
    target_col="label",
    processors=[
        ProcessorConfig("tabular", {"normalize_numerical": True}),
        ProcessorConfig("custom", {"param1": "value1"})
    ]
)
```

### 2. Data Loading

The `TabularDataModule` handles data loading and preprocessing:

```python
from tabular_ssl.data.datamodule import TabularDataModule

datamodule = TabularDataModule(
    data_dir="data",
    train_file="train.parquet",
    val_file="val.parquet",
    test_file="test.parquet",
    feature_config=feature_config,
    batch_size=32,
    sequence_length=100
)
```

### 3. Model Architecture

The framework supports multiple sequence encoder types:

```python
# Configuration for different model types
model_config = {
    "event_encoder": {
        "input_dim": 10,
        "hidden_dims": [64, 32],
        "output_dim": 16
    },
    "sequence_encoder": {
        "model_type": "transformer",  # or "rnn", "lstm", "gru", "ssm", "s4"
        "input_dim": 16,
        "hidden_dim": 32,
        "num_layers": 2
    }
}
```

## Usage

### 1. Basic Training

```bash
python src/train.py
```

### 2. Custom Configuration

```bash
python src/train.py model.sequence_encoder.model_type=s4 trainer.max_epochs=200
```

### 3. Debug Mode

```bash
python src/train.py debug=true
```

## Custom Feature Processors

You can create custom feature processors by extending the `FeatureProcessor` class:

```python
from tabular_ssl.data.datamodule import FeatureProcessor, FeatureProcessorRegistry

@FeatureProcessorRegistry.register("custom")
class CustomFeatureProcessor(FeatureProcessor):
    def __init__(self, config: FeatureConfig):
        self.config = config
        self._feature_dims = None
        self._processed_data = None
    
    def fit(self, data: pl.DataFrame) -> None:
        # Custom fitting logic
        self._processed_data = data
        self._feature_dims = {"custom": 10}
    
    def transform(self, data: pl.DataFrame) -> Dict[str, torch.Tensor]:
        # Custom transformation logic
        return {"custom": torch.randn(len(data), 10)}
    
    def get_feature_dims(self) -> Dict[str, int]:
        return self._feature_dims
```

## Configuration

The project uses Hydra for configuration management. Key configuration files:

### 1. Main Configuration (`configs/config.yaml`)
```yaml
defaults:
  - _self_
  - model: default
  - data: default
  - trainer: default
```

### 2. Model Configuration (`configs/model/default.yaml`)
```yaml
event_encoder:
  input_dim: 10
  hidden_dims: [64, 32]
  output_dim: 16

sequence_encoder:
  model_type: "transformer"
  input_dim: 16
  hidden_dim: 32
```

### 3. Data Configuration (`configs/data/default.yaml`)
```yaml
data_dir: ${paths.data_dir}
train_file: train.parquet
val_file: val.parquet
test_file: test.parquet
batch_size: 32
sequence_length: 100
```

## Sample Data

The project includes a sample financial transaction dataset generator that creates realistic transaction sequences. The data includes:

### Entity Features
- Entity ID
- Age
- Income level
- Credit score
- Account age
- Risk level

### Transaction Features
- Transaction ID
- Timestamp
- Amount
- Category (groceries, dining, transportation, etc.)
- Merchant type
- Location
- Status
- Fraud indicator

### Data Generation

Generate sample data using the CLI:

```bash
# Generate default sample data
python -m tabular_ssl.cli download-data

# Customize data generation
python -m tabular_ssl.cli download-data \
    --output-dir data/custom \
    --n-entities 2000 \
    --n-transactions 200000 \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --seed 42
```

The generator creates realistic patterns in the data:
- More transactions during business hours and weekdays
- Different transaction amount distributions per category
- Realistic entity transaction frequencies
- Chronological train/val/test splits

### Using Sample Data

The generated data is saved in Parquet format with the following structure:

```
data/sample/
├── entities.parquet      # Entity features
├── transactions.parquet  # All transactions
├── train.parquet        # Training set (70%)
├── val.parquet          # Validation set (15%)
└── test.parquet         # Test set (15%)
```

To use the sample data in your experiments:

```python
from tabular_ssl.data.datamodule import TabularDataModule

# Initialize data module with sample data
datamodule = TabularDataModule(
    data_dir="data/sample",
    train_file="train.parquet",
    val_file="val.parquet",
    test_file="test.parquet",
    batch_size=32,
    sequence_length=100
)
```

## End-to-End Guide

### 1. Setup and Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tabular-ssl.git
cd tabular-ssl

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Sample Data

```bash
# Generate default sample data (1000 entities, 100K transactions)
python -m tabular_ssl.cli download-data

# Generate custom dataset
python -m tabular_ssl.cli download-data \
    --output-dir data/custom \
    --n-entities 2000 \
    --n-transactions 200000 \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --seed 42
```

The generated data will be saved in Parquet format:
```
data/sample/
├── entities.parquet      # Entity features
├── transactions.parquet  # All transactions
├── train.parquet        # Training set (70%)
├── val.parquet          # Validation set (15%)
└── test.parquet         # Test set (15%)
```

### 3. Quick Start with Default Configuration

```bash
# Train with default configuration
python src/train.py

# Train with debug mode
python src/train.py debug=true
```

### 4. Experimenting with Different Models

#### 4.1 Using Pre-defined Configurations

```bash
# Train with S4 model
python src/train.py experiment=s4_large

# Train with Transformer model
python src/train.py experiment=transformer_small
```

#### 4.2 Customizing via Command Line

```bash
# Change sequence encoder
python src/train.py model.sequence_encoder.model_type=transformer

# Modify model parameters
python src/train.py \
    model.sequence_encoder.hidden_dim=128 \
    model.sequence_encoder.num_layers=4 \
    model.event_encoder.hidden_dims=[256,128,64]

# Adjust training parameters
python src/train.py \
    trainer.max_epochs=100 \
    trainer.accelerator=gpu \
    trainer.devices=2 \
    trainer.precision=16-mixed
```

#### 4.3 Creating Custom Configurations

Create a new configuration file in `configs/experiments/`:

```yaml
# configs/experiments/custom_model.yaml
model:
  sequence_encoder:
    model_type: "s4"
    hidden_dim: 256
    num_layers: 6
    state_dim: 128
    max_sequence_length: 2048

  event_encoder:
    hidden_dims: [512, 256, 128]
    dropout: 0.2

trainer:
  max_epochs: 200
  precision: "16-mixed"
```

Run with custom configuration:
```bash
python src/train.py experiment=custom_model
```

### 5. Data Loading and Preprocessing

#### 5.1 Basic Usage

```python
from tabular_ssl.data.datamodule import TabularDataModule

# Initialize data module
datamodule = TabularDataModule(
    data_dir="data/sample",
    train_file="train.parquet",
    val_file="val.parquet",
    test_file="test.parquet",
    batch_size=32,
    sequence_length=100
)

# Setup data
datamodule.setup()

# Get training dataloader
train_loader = datamodule.train_dataloader()
```

#### 5.2 Custom Feature Processing

```python
from tabular_ssl.data.datamodule import FeatureConfig, ProcessorConfig

# Define feature configuration
feature_config = FeatureConfig(
    categorical_cols=["category", "merchant_type", "location"],
    numerical_cols=["amount"],
    target_col="is_fraud",
    processors=[
        ProcessorConfig("tabular", {"normalize_numerical": True}),
        ProcessorConfig("custom", {"param1": "value1"})
    ]
)

# Initialize data module with custom features
datamodule = TabularDataModule(
    data_dir="data/sample",
    feature_config=feature_config,
    batch_size=32,
    sequence_length=100
)
```

### 6. Running Parameter Sweeps

```bash
# Sweep over different sequence encoders
python src/train.py -m model.sequence_encoder.model_type=transformer,lstm,s4

# Sweep over multiple parameters
python src/train.py -m \
    model.sequence_encoder.model_type=transformer,s4 \
    model.sequence_encoder.hidden_dim=32,64,128 \
    trainer.max_epochs=50,100
```

### 7. Experiment Tracking

The framework integrates with various experiment trackers. Configure in `configs/logger/default.yaml`:

```yaml
default:
  - _self_
  - wandb: true  # or mlflow, tensorboard
  - wandb:
      project: "tabular-ssl"
      name: ${now:%Y-%m-%d_%H-%M-%S}
      tags: ${model.sequence_encoder.model_type}
```

### 8. Common Use Cases

#### 8.1 Fraud Detection

```bash
# Train model for fraud detection
python src/train.py \
    model.prediction_head.num_classes=2 \
    model.prediction_head.task_type=binary_classification \
    trainer.max_epochs=100
```

#### 8.2 Transaction Classification

```bash
# Train model for transaction category prediction
python src/train.py \
    model.prediction_head.num_classes=6 \
    model.prediction_head.task_type=multiclass_classification \
    trainer.max_epochs=100
```

#### 8.3 Anomaly Detection

```bash
# Train model for anomaly detection
python src/train.py \
    model.prediction_head.task_type=reconstruction \
    trainer.max_epochs=100
```

### 9. Troubleshooting

Common issues and solutions:

1. **Out of Memory**
   - Reduce batch size: `trainer.batch_size=16`
   - Use gradient accumulation: `trainer.accumulate_grad_batches=2`
   - Enable mixed precision: `trainer.precision=16-mixed`

2. **Slow Training**
   - Increase batch size if memory allows
   - Reduce sequence length
   - Use fewer model parameters
   - Enable mixed precision training

3. **Poor Performance**
   - Try different sequence encoders
   - Adjust model architecture
   - Tune learning rate
   - Increase training epochs

4. **Data Loading Issues**
   - Check file paths
   - Verify Parquet file format
   - Ensure correct feature configuration
   - Check data preprocessing steps

## Contributing

This project was created through AI-assisted development. Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was created through an AI chat conversation with Cursor
- Built on PyTorch Lightning and Hydra
- Inspired by various self-supervised learning approaches for tabular data 

## Quick Reference

### Data Generation
```bash
# Generate default sample data
python -m tabular_ssl.cli download-data

# Generate custom dataset
python -m tabular_ssl.cli download-data --n-entities 2000 --n-transactions 200000
```

### Training Commands
```bash
# Basic training
python src/train.py

# Debug mode
python src/train.py debug=true

# Train with specific model
python src/train.py experiment=s4_large
python src/train.py experiment=transformer_small
```

### Model Configuration
```bash
# Change sequence encoder
python src/train.py model.sequence_encoder.model_type=transformer
python src/train.py model.sequence_encoder.model_type=s4
python src/train.py model.sequence_encoder.model_type=lstm

# Modify model architecture
python src/train.py model.sequence_encoder.hidden_dim=128 model.sequence_encoder.num_layers=4
python src/train.py model.event_encoder.hidden_dims=[256,128,64]
```

### Training Parameters
```bash
# Basic training parameters
python src/train.py trainer.max_epochs=100 trainer.batch_size=64

# GPU training
python src/train.py trainer.accelerator=gpu trainer.devices=2

# Mixed precision
python src/train.py trainer.precision=16-mixed

# Gradient accumulation
python src/train.py trainer.accumulate_grad_batches=2
```

### Experiment Tracking
```bash
# Enable WandB logging
python src/train.py logger=wandb

# Custom experiment name
python src/train.py logger.wandb.name=my_experiment

# Add tags
python src/train.py logger.wandb.tags=[s4,large]
```

### Parameter Sweeps
```bash
# Sweep over models
python src/train.py -m model.sequence_encoder.model_type=transformer,lstm,s4

# Sweep over multiple parameters
python src/train.py -m \
    model.sequence_encoder.model_type=transformer,s4 \
    model.sequence_encoder.hidden_dim=32,64,128
```

### Common Tasks
```bash
# Fraud detection
python src/train.py model.prediction_head.task_type=binary_classification

# Transaction classification
python src/train.py model.prediction_head.task_type=multiclass_classification

# Anomaly detection
python src/train.py model.prediction_head.task_type=reconstruction
```

### Data Loading
```bash
# Custom data directory
python src/train.py data.data_dir=data/custom

# Modify sequence length
python src/train.py data.sequence_length=200

# Change batch size
python src/train.py data.batch_size=64
```

### Feature Processing
```bash
# Enable numerical normalization
python src/train.py data.normalize_numerical=true

# Change categorical encoding
python src/train.py data.categorical_encoding=embedding
```

### Common Overrides
```bash
# Override multiple parameters
python src/train.py \
    model.sequence_encoder.model_type=s4 \
    model.sequence_encoder.hidden_dim=128 \
    trainer.max_epochs=100 \
    trainer.precision=16-mixed \
    data.batch_size=64 \
    logger.wandb.name=my_s4_experiment
```

### Debugging
```bash
# Enable debug mode
python src/train.py debug=true

# Print configuration
python src/train.py print_config=true

# Ignore warnings
python src/train.py ignore_warnings=true
```

### Input Corruption Strategies

The framework provides several input corruption strategies for self-supervised learning tasks, inspired by ReConTab, SCARF, and VIME:

#### 1. Random Masking (SCARF-style)
```python
from tabular_ssl.models import RandomMasking

# Basic masking
corruption = RandomMasking({
    "corruption_rate": 0.15,
    "mask_value": 0.0
})

# Importance-based masking
corruption = RandomMasking({
    "corruption_rate": 0.15,
    "mask_distribution": "importance",
    "feature_importance": [0.1, 0.2, 0.3, 0.4]  # Relative importance of features
})

# Apply corruption
x_corrupted = corruption(x)
```

#### 2. Gaussian Noise
```python
from tabular_ssl.models import GaussianNoise

# Feature-wise noise
corruption = GaussianNoise({
    "noise_scale": 0.1,
    "feature_wise": True
})

# Global noise
corruption = GaussianNoise({
    "noise_scale": 0.1,
    "feature_wise": False
})

# Apply corruption
x_corrupted = corruption(x)
```

#### 3. Feature Swapping (ReConTab-style)
```python
from tabular_ssl.models import SwappingCorruption

# Random swapping
corruption = SwappingCorruption({
    "swap_prob": 0.1
})

# Group-based swapping
corruption = SwappingCorruption({
    "swap_prob": 0.1,
    "feature_groups": [
        [0, 1, 2],  # First group of features
        [3, 4, 5]   # Second group of features
    ]
})

# Apply corruption
x_corrupted = corruption(x)
```

#### 4. VIME-style Corruption
```python
from tabular_ssl.models import VIMECorruption

corruption = VIMECorruption({
    "mask_prob": 0.15,
    "swap_prob": 0.1,
    "noise_scale": 0.1,
    "feature_importance": [0.1, 0.2, 0.3, 0.4]  # Optional
})

# Apply corruption
x_corrupted = corruption(x)
```

#### 5. Corruption Pipeline
Combine multiple corruption strategies:
```python
from tabular_ssl.models import CorruptionPipeline

# Create a pipeline with multiple corruptions
corruption = CorruptionPipeline({
    "use_masking": True,
    "use_noise": True,
    "use_swapping": True,
    "use_vime": False,
    "corruption_rate": 0.15,
    "noise_scale": 0.1,
    "swap_prob": 0.1
})

# Apply corruption pipeline
x_corrupted = corruption(x)
```

### Using Corruption with AutoEncoder

```python
from tabular_ssl.models import AutoEncoderEventEncoder, CorruptionPipeline

# Create corruption pipeline
corruption = CorruptionPipeline({
    "use_masking": True,
    "use_noise": True,
    "corruption_rate": 0.15,
    "noise_scale": 0.1
})

# Create autoencoder
encoder = AutoEncoderEventEncoder({
    "input_dim": 10,
    "hidden_dims": [64, 32],
    "output_dim": 16,
    "dropout": 0.1
})

# Training step
def training_step(x):
    # Corrupt input
    x_corrupted = corruption(x)
    
    # Encode corrupted input
    z = encoder(x_corrupted)
    
    # Reconstruct original input
    x_recon = encoder.decode(z)
    
    # Compute reconstruction loss
    loss = encoder.reconstruction_loss(x, x_recon)
    
    return loss
```

### Using Corruption with Contrastive Learning

```python
from tabular_ssl.models import ContrastiveEventEncoder, CorruptionPipeline

# Create two different corruption pipelines
corruption1 = CorruptionPipeline({
    "use_masking": True,
    "corruption_rate": 0.15
})

corruption2 = CorruptionPipeline({
    "use_noise": True,
    "noise_scale": 0.1
})

# Create contrastive encoder
encoder = ContrastiveEventEncoder({
    "input_dim": 10,
    "hidden_dims": [64, 32],
    "output_dim": 16,
    "dropout": 0.1,
    "temperature": 0.07
})

# Training step
def training_step(x):
    # Create two different views
    x1 = corruption1(x)
    x2 = corruption2(x)
    
    # Get contrastive representations
    z1 = encoder(x1)
    z2 = encoder(x2)
    
    # Compute contrastive loss
    loss = encoder.contrastive_loss(z1, z2)
    
    return loss
```

### Configuration Examples

#### 1. Basic Corruption
```yaml
model:
  corruption:
    _target_: tabular_ssl.models.components.RandomMasking
    corruption_rate: 0.15
    mask_value: 0.0
```

#### 2. VIME-style Corruption
```yaml
model:
  corruption:
    _target_: tabular_ssl.models.components.VIMECorruption
    mask_prob: 0.15
    swap_prob: 0.1
    noise_scale: 0.1
    feature_importance: [0.1, 0.2, 0.3, 0.4]
```

#### 3. Corruption Pipeline
```yaml
model:
  corruption:
    _target_: tabular_ssl.models.components.CorruptionPipeline
    use_masking: true
    use_noise: true
    use_swapping: true
    use_vime: false
    corruption_rate: 0.15
    noise_scale: 0.1
    swap_prob: 0.1
    feature_groups:
      - [0, 1, 2]
      - [3, 4, 5]
``` 