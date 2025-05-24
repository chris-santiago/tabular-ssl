# Tabular SSL Configuration System

This directory contains the cleaned up and reorganized configuration system for the tabular-ssl project. The configs are designed for modularity, extensibility, and compatibility with our component-based architecture.

## Directory Structure

```
configs/
├── config.yaml              # Main configuration file
├── README.md                 # This documentation
├── data/                     # Data module configurations
│   ├── default.yaml         # Standard dataset config
│   └── simple.yaml          # Simple/test dataset config
├── model/                    # Model component configurations
│   ├── default.yaml         # Default model assembly
│   ├── event_encoder/       # Event encoder components
│   │   └── mlp.yaml        # MLP event encoder
│   ├── sequence_encoder/    # Sequence encoder components
│   │   ├── transformer.yaml # Transformer encoder
│   │   ├── s4.yaml         # S4 encoder
│   │   ├── rnn.yaml        # RNN/LSTM encoder
│   │   └── null.yaml       # No sequence processing
│   ├── embedding/           # Embedding layer configs
│   │   └── categorical.yaml # Categorical embeddings
│   ├── projection_head/     # Projection head configs
│   │   └── mlp.yaml        # MLP projection head
│   └── prediction_head/     # Prediction head configs
│       └── classification.yaml # Classification head
├── experiments/             # Pre-configured experiments
│   ├── simple_mlp.yaml     # MLP-only baseline
│   ├── transformer_small.yaml # Small transformer
│   ├── s4_large.yaml       # Large S4 model
│   └── rnn_baseline.yaml   # RNN baseline
├── trainer/                 # PyTorch Lightning trainer configs
├── callbacks/               # Training callbacks
├── logger/                  # Logging configurations
│   ├── wandb.yaml          # Weights & Biases
│   └── csv.yaml            # CSV logging
├── paths/                   # Path configurations
├── hydra/                   # Hydra framework settings
└── extras/                  # Extra utilities
```

## Key Features

### 🧩 **Modular Architecture**
- Each component (event encoder, sequence encoder, etc.) has its own config
- Easy to mix and match components
- Clean separation of concerns

### 🔧 **Extensible Design**
- Add new components by creating new config files
- Override specific parameters in experiments
- Support for custom architectures

### ⚡ **Ready-to-Use Experiments**
- Pre-configured experiments for common architectures
- Optimized hyperparameters for each model type
- Easy to run with single command

## Usage Examples

### Basic Training
```bash
# Train with default configuration
python train.py

# Train with specific data config
python train.py data=simple

# Train with specific logger
python train.py logger=wandb

# Train with CSV logging for local development
python train.py logger=csv
```

### Experiment Configurations
```bash
# Run MLP-only baseline
python train.py +experiment=simple_mlp

# Run small transformer experiment
python train.py +experiment=transformer_small

# Run large S4 experiment
python train.py +experiment=s4_large

# Run RNN baseline
python train.py +experiment=rnn_baseline
```

### Custom Component Combinations
```bash
# MLP event encoder + Transformer sequence encoder
python train.py model/event_encoder=mlp model/sequence_encoder=transformer

# MLP event encoder only (no sequence processing)
python train.py model/sequence_encoder=null

# RNN sequence encoder with custom settings
python train.py model/sequence_encoder=rnn

# Custom learning rate and batch size
python train.py model.learning_rate=1e-4 data.batch_size=128
```

## Component Configurations

### Event Encoders
- **MLP**: Multi-layer perceptron (64→128→256→512) with batch norm and ReLU

### Sequence Encoders
- **Transformer**: Self-attention based (4 layers, 8 heads, 512 hidden)
- **S4**: Structured state space model (2 layers, 64 hidden, bidirectional)
- **RNN**: LSTM/GRU encoder (2 layers, 128 hidden)
- **null**: No sequence processing (MLP-only)

### Heads
- **Projection Head**: MLP for representation projection
- **Prediction Head**: Classification head with dropout

### Embeddings
- **Categorical**: Flexible embedding dimensions per categorical feature

## Default Model Configuration

The default model uses:
- **Event Encoder**: MLP (64→128→256→512)
- **Sequence Encoder**: Transformer (4 layers, 8 heads)
- **Embedding**: Categorical embeddings for features
- **Projection Head**: MLP (512→256→128)
- **Prediction Head**: Classification (128→64→2)
- **Training**: AdamW optimizer, cosine scheduler

## Experiment Configurations

### Simple MLP (`simple_mlp`)
- **Architecture**: MLP-only (32→64→128→256)
- **Components**: Event encoder only, no sequence processing
- **Best for**: Quick baselines, simple datasets
- **Training**: 50 epochs, batch size 128, AdamW optimizer

### Transformer Small (`transformer_small`)
- **Architecture**: MLP (32→64→128) + Transformer (2 layers, 4 heads)
- **Components**: Small transformer for sequence modeling
- **Best for**: Medium-length sequences, attention patterns
- **Training**: 50 epochs, mixed precision, cosine scheduler

### S4 Large (`s4_large`)
- **Architecture**: MLP (64→128→256→512) + S4 (6 layers, 512 hidden)
- **Components**: Large S4 model for long sequences
- **Best for**: Long sequences, efficient processing
- **Training**: 100 epochs, gradient accumulation, mixed precision

### RNN Baseline (`rnn_baseline`)
- **Architecture**: MLP (32→64→128) + Bidirectional LSTM (2 layers)
- **Components**: LSTM sequence encoder with 256 output (128×2)
- **Best for**: Sequential patterns, baseline comparison
- **Training**: 50 epochs, step scheduler, gradient clipping

## Data Configurations

### Default (`default`)
- **Dataset**: sample_dataset with 2 categorical + 8 numerical features
- **Sequences**: Length 32, 2-100 events per sequence
- **DataLoader**: Batch size 64, 4 workers, pin memory
- **Split**: 70% train, 15% validation, 15% test

### Simple (`simple`)
- **Dataset**: simple_test with 1 categorical + 4 numerical features
- **Sequences**: Length 16, 1-50 events per sequence
- **DataLoader**: Batch size 32, 2 workers
- **Split**: 80% train, 10% validation, 10% test

## Configuration Philosophy

### ✅ **What We Did**
1. **Simplified Structure**: Removed redundant configs and complex interpolations
2. **Component-Based**: Each model component has its own config file
3. **Experiment-Driven**: Pre-configured experiments for common use cases
4. **Hydra Compatible**: Proper `_target_` specifications for all components
5. **Modular Overrides**: Easy to override specific parameters

### ❌ **What We Removed**
1. **Legacy Patterns**: Old config structures that don't match new architecture
2. **Complex Interpolations**: Simplified variable references
3. **Duplicate Directories**: Consolidated `experiment/` and `experiments/`
4. **Redundant Settings**: Removed duplicate or unused configuration options
5. **Over-Engineering**: Simplified to focus on actual use cases

## Adding New Components

### 1. Create Component Config
```yaml
# configs/model/sequence_encoder/my_encoder.yaml
_target_: tabular_ssl.models.components.MySequenceEncoder
input_dim: 128
hidden_dim: 256
custom_param: value
```

### 2. Create Experiment Config
```yaml
# configs/experiments/my_experiment.yaml
# @package _global_
defaults:
  - override /model/sequence_encoder: my_encoder

tags: ["custom", "experiment"]

model:
  sequence_encoder:
    custom_param: optimized_value
```

### 3. Use in Training
```bash
python train.py +experiment=my_experiment
```

## Configuration Details

### Model Component Parameters

**Event Encoder (MLP)**:
```yaml
input_dim: 64          # Input feature dimension
hidden_dims: [128, 256] # Hidden layer sizes
output_dim: 512        # Output dimension
dropout: 0.1           # Dropout rate
activation: relu       # Activation function
use_batch_norm: true   # Use batch normalization
```

**Sequence Encoder (Transformer)**:
```yaml
input_dim: 512         # Input dimension (from event encoder)
hidden_dim: 512        # Hidden dimension
num_layers: 4          # Number of transformer layers
num_heads: 8           # Number of attention heads
dim_feedforward: 2048  # Feedforward dimension
dropout: 0.1           # Dropout rate
max_seq_length: 2048   # Maximum sequence length
```

**Training Configuration**:
```yaml
learning_rate: 1.0e-4  # Learning rate
weight_decay: 0.01     # Weight decay
optimizer_type: adamw  # Optimizer (adamw, adam, sgd)
scheduler_type: cosine # Scheduler (cosine, step, plateau, null)
```

## Best Practices

1. **Start with Experiments**: Use pre-configured experiments as starting points
2. **Override Incrementally**: Make small changes to existing configs
3. **Test Components**: Verify new components work before creating experiments
4. **Document Changes**: Update this README when adding new configurations
5. **Use Tags**: Tag experiments for easy filtering and organization
6. **Check Dimensions**: Ensure input/output dimensions match between components

## Troubleshooting

### Common Issues
1. **Missing `_target_`**: All component configs need `_target_` specification
2. **Wrong Override Syntax**: Use `+experiment=name` for experiments
3. **Dimension Mismatches**: Check input/output dimensions between components
4. **Import Errors**: Ensure `PYTHONPATH` includes `src/` directory
5. **Null Sequence Encoder**: Use `model/sequence_encoder=null` for MLP-only

### Validation Commands
```bash
# Test main config loading
python -c "import hydra; hydra.initialize(config_path='configs', version_base=None); cfg = hydra.compose(config_name='config'); print('✅ Main config valid')"

# Test experiment loading
python -c "import hydra; hydra.initialize(config_path='configs', version_base=None); cfg = hydra.compose(config_name='config', overrides=['+experiment=simple_mlp']); print('✅ Experiment config valid')"

# Test component override
python -c "import hydra; hydra.initialize(config_path='configs', version_base=None); cfg = hydra.compose(config_name='config', overrides=['model/sequence_encoder=null']); print('✅ Component override valid')"
```

### Debugging Tips
```bash
# Print full config
python train.py --config-name=config --print-config

# Print experiment config
python train.py +experiment=simple_mlp --print-config

# Dry run (no training)
python train.py +experiment=simple_mlp train=false test=false

# Use simple data for quick testing
python train.py data=simple +experiment=simple_mlp
```

## Environment Setup

Ensure proper environment setup:

```bash
# Set PYTHONPATH for imports
export PYTHONPATH=/path/to/tabular-ssl/src

# Optional: Set PROJECT_ROOT for data paths
export PROJECT_ROOT=/path/to/tabular-ssl
```

# Configurations

This directory contains Hydra configuration files for the Tabular SSL library.

## Structure

```
configs/
├── config.yaml                 # Main configuration file
├── data/                       # Data configurations
│   ├── synthetic.yaml          # Synthetic tabular data
│   ├── credit_card.yaml        # Credit card transactions (IBM TabFormer)
│   └── custom.yaml             # Template for custom data
├── model/                      # Model component configurations
│   ├── event_encoder/          # Event encoding architectures
│   ├── sequence_encoder/       # Sequence modeling architectures
│   ├── projection_head/        # Projection head configurations
│   └── prediction_head/        # Prediction head configurations
├── experiments/                # Complete experiment configurations
│   ├── simple_demo.yaml        # Basic demonstration
│   ├── credit_card_demo.yaml   # Credit card data demo
│   └── transformer_ssl.yaml    # Advanced transformer setup
├── trainer/                    # PyTorch Lightning trainer settings
├── callbacks/                  # Training callbacks (checkpointing, logging)
├── logger/                     # Experiment logging configurations
└── paths/                      # Directory paths
```

## Quick Start

### Using Sample Data (Credit Card Transactions)

The easiest way to get started is with the IBM TabFormer credit card dataset:

```bash
# Run the demo to download and explore the data
python demo_credit_card_data.py

# Train a model on credit card transactions
python train.py +experiment=credit_card_demo
```

The credit card dataset provides:
- **Sequential transaction data** from real credit card usage patterns
- **Mixed data types**: categorical (merchant, category) and numerical (amount, time)
- **Multiple users**: sequences of transactions per credit card holder
- **Realistic patterns**: temporal dependencies and user behavior

### Using Your Own Data

To use custom data, create a new data configuration:

```yaml
# configs/data/my_data.yaml
_target_: tabular_ssl.data.datamodule.TabularDataModule

data_dir: ${paths.data_dir}
train_file: "my_train.parquet"
val_file: "my_val.parquet" 
test_file: "my_test.parquet"

sequence_length: 50
batch_size: 32

feature_config:
  categorical_cols: ["category", "merchant"]
  numerical_cols: ["amount", "time_since_last"]
  normalize_numerical: true
  categorical_encoding: embedding
```

## Sample Data Sources

### Credit Card Transactions (IBM TabFormer)

- **Source**: [IBM TabFormer Repository](https://github.com/IBM/TabFormer/blob/main/data/credit_card/transactions.tgz)
- **Description**: Real credit card transaction sequences with privacy protection
- **Features**: Transaction amounts, merchant categories, timestamps, user IDs
- **Use cases**: Fraud detection, spending pattern analysis, sequence prediction

**Configuration**: `configs/data/credit_card.yaml`

```yaml
_target_: tabular_ssl.data.datamodule.TabularDataModule

use_sample_data: true
sample_data_config:
  data_source: "credit_card" 
  n_users: 1000
  sequence_length: 32

sequence_length: 32
batch_size: 64
```

## Experiment Templates

### Credit Card Demo

A complete experiment using credit card transaction data:

```bash
python train.py +experiment=credit_card_demo
```

This experiment:
- Downloads IBM TabFormer credit card data automatically
- Uses a transformer-based sequence encoder
- Includes proper validation and checkpointing
- Optimized for real transaction patterns

### Custom Experiment Template

```yaml
# configs/experiments/my_experiment.yaml
# @package _global_

defaults:
  - override /data: my_data
  - override /model/event_encoder: mlp
  - override /model/sequence_encoder: transformer

model:
  learning_rate: 1e-4
  sequence_encoder:
    num_layers: 6
    num_heads: 8

trainer:
  max_epochs: 50
  precision: 16-mixed
```

## Configuration Components

### Event Encoders (`model/event_encoder/`)

Transform individual events/transactions into dense representations:

- **MLP**: Multi-layer perceptron for mixed tabular features
- **Embedding**: Specialized for categorical-heavy data
- **Linear**: Simple linear transformation

### Sequence Encoders (`model/sequence_encoder/`)

Model temporal dependencies between events:

- **Transformer**: Self-attention for complex dependencies
- **LSTM**: Recurrent networks for sequential patterns
- **GRU**: Lighter alternative to LSTM
- **1D CNN**: Convolutional approach for local patterns

### Model Architecture

The complete model pipeline:

```
Raw Tabular Data
       ↓
Event Encoder (per transaction)
       ↓
Sequence Encoder (across time)
       ↓
Projection Head
       ↓
Task-specific Head
```

## Data Configuration Details

### Feature Processing

```yaml
feature_config:
  # Columns to treat as categorical
  categorical_cols: ["merchant_category", "transaction_type"]
  
  # Columns to treat as numerical  
  numerical_cols: ["amount", "time_since_last"]
  
  # Target column (optional)
  target_col: "is_fraud"
  
  # How to encode categorical features
  categorical_encoding: "embedding"  # or "onehot"
  
  # Whether to normalize numerical features
  normalize_numerical: true
  
  # Custom embedding dimensions per categorical column
  categorical_embedding_dims:
    merchant_category: 32
    transaction_type: 8
```

### Sequence Settings

```yaml
# Length of each sequence
sequence_length: 32

# Filter users by transaction count
filter_users:
  min_transactions: 10
  max_transactions: 200

# Data splitting
train_val_test_split: [0.7, 0.15, 0.15]
```

## Advanced Usage

### Multiple Data Sources

```yaml
# Combine different data sources
data:
  primary_source: "credit_card"
  auxiliary_sources: ["user_profiles", "merchant_data"]
  
sample_data_config:
  data_source: "credit_card"
  n_users: 5000
  include_fraud_labels: true
```

### Custom Preprocessing

```yaml
# Custom preprocessing pipeline
preprocessing:
  - normalize_amounts
  - encode_timestamps  
  - create_user_features
  - filter_outliers
```

### Distributed Training

```yaml
trainer:
  strategy: "ddp"  # Distributed data parallel
  devices: 4
  num_nodes: 1
  precision: 16-mixed
```

## Validation Commands

Test your configurations:

```bash
# Validate data configuration
python train.py data=credit_card --cfg job --resolve

# Test data loading
python -c "
from tabular_ssl.data.datamodule import TabularDataModule
dm = TabularDataModule(use_sample_data=True, sample_data_config={'data_source': 'credit_card'})
dm.prepare_data()
dm.setup()
print('✅ Data configuration valid')
"

# Run quick training test
python train.py +experiment=credit_card_demo trainer.fast_dev_run=5
```

## Contributing

When adding new configurations:

1. **Data configs**: Add to `data/` with clear documentation
2. **Model configs**: Split into logical components under `model/`
3. **Experiments**: Include complete working examples in `experiments/`
4. **Documentation**: Update this README with usage examples

For sample data:
1. Add download function to `src/tabular_ssl/data/sample_data.py`
2. Create data config in `configs/data/`
3. Add demo experiment in `configs/experiments/`
4. Update documentation with examples

## Troubleshooting

**Data download issues**:
```bash
# Force re-download
python -c "from tabular_ssl.data.sample_data import download_credit_card_transactions; download_credit_card_transactions(force_download=True)"
```

**Memory issues with large datasets**:
```yaml
data:
  batch_size: 16  # Reduce batch size
  num_workers: 2  # Reduce workers
  
model:
  sequence_encoder:
    hidden_dim: 256  # Reduce model size
```

**Feature detection issues**:
```yaml
feature_config:
  auto_detect: false  # Disable auto-detection
  categorical_cols: ["col1", "col2"]  # Specify manually
  numerical_cols: ["col3", "col4"]
``` 