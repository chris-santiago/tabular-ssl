# Tabular SSL Configuration System

This directory contains the cleaned up and reorganized configuration system for the tabular-ssl project. The configs are designed for modularity, extensibility, and compatibility with our component-based architecture.

## Directory Structure

```
configs/
├── config.yaml              # Main configuration file
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

# Custom learning rate and batch size
python train.py model.learning_rate=1e-4 data.batch_size=128
```

## Component Configurations

### Event Encoders
- **MLP**: Multi-layer perceptron for basic feature encoding
- **Tabular**: Specialized for categorical + numerical features (built into model)

### Sequence Encoders
- **Transformer**: Self-attention based sequence modeling
- **S4**: Structured state space model for long sequences
- **RNN**: LSTM/GRU for sequential processing
- **null**: No sequence processing (MLP-only)

### Heads
- **Projection Head**: Projects representations to different dimensions
- **Prediction Head**: Final classification/regression layer

## Experiment Configurations

### Simple MLP (`simple_mlp`)
- Event encoder: MLP (32→64→128→256)
- Sequence encoder: None
- Best for: Quick baselines, simple datasets
- Training: 50 epochs, batch size 128

### Transformer Small (`transformer_small`)
- Event encoder: MLP (32→64→128)
- Sequence encoder: Transformer (2 layers, 4 heads)
- Best for: Medium-length sequences, attention patterns
- Training: 50 epochs, mixed precision

### S4 Large (`s4_large`)
- Event encoder: MLP (64→128→256→512)
- Sequence encoder: S4 (6 layers, 512 hidden)
- Best for: Long sequences, efficient processing
- Training: 100 epochs, gradient accumulation

### RNN Baseline (`rnn_baseline`)
- Event encoder: MLP (32→64→128)
- Sequence encoder: Bidirectional LSTM (2 layers)
- Best for: Sequential patterns, baseline comparison
- Training: 50 epochs, standard settings

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

## Best Practices

1. **Start with Experiments**: Use pre-configured experiments as starting points
2. **Override Incrementally**: Make small changes to existing configs
3. **Test Components**: Verify new components work before creating experiments
4. **Document Changes**: Update this README when adding new configurations
5. **Use Tags**: Tag experiments for easy filtering and organization

## Troubleshooting

### Common Issues
1. **Missing `_target_`**: All component configs need `_target_` specification
2. **Wrong Override Syntax**: Use `+experiment=name` for experiments
3. **Dimension Mismatches**: Check input/output dimensions between components
4. **Import Errors**: Ensure `PYTHONPATH` includes `src/` directory

### Validation
```bash
# Test config loading
python -c "import hydra; hydra.initialize(config_path='configs'); cfg = hydra.compose(config_name='config'); print('✅ Config valid')"

# Test experiment loading
python -c "import hydra; hydra.initialize(config_path='configs'); cfg = hydra.compose(config_name='config', overrides=['+experiment=simple_mlp']); print('✅ Experiment valid')"
``` 