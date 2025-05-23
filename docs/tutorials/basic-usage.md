# Basic Usage

This tutorial covers common use cases and patterns for working with Tabular SSL.

## Working with Different Data Types

### Numerical Data

```python
import pandas as pd
from tabular_ssl import TabularSSL
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Create sample numerical data
data = pd.DataFrame({
    'feature1': [1.0, 2.0, 3.0],
    'feature2': [4.0, 5.0, 6.0]
})

# Create configuration
config = OmegaConf.create({
    '_target_': 'tabular_ssl.models.TabularSSL',
    'input_dim': 2
})

# Initialize and train model
model = instantiate(config)
model.train(data)
```

### Categorical Data

```python
# Create sample categorical data
data = pd.DataFrame({
    'category1': ['A', 'B', 'A'],
    'category2': ['X', 'Y', 'Z']
})

# Preprocess categorical data
from tabular_ssl.data import DataLoader
data_loader = DataLoader()
processed_data = data_loader.preprocess(data, categorical_cols=['category1', 'category2'])

# Create configuration
config = OmegaConf.create({
    '_target_': 'tabular_ssl.models.TabularSSL',
    'input_dim': processed_data.shape[1]
})

# Train model
model = instantiate(config)
model.train(processed_data)
```

### Mixed Data Types

```python
# Create sample mixed data
data = pd.DataFrame({
    'numeric': [1.0, 2.0, 3.0],
    'category': ['A', 'B', 'A']
})

# Preprocess data
processed_data = data_loader.preprocess(
    data,
    categorical_cols=['category']
)

# Create configuration
config = OmegaConf.create({
    '_target_': 'tabular_ssl.models.TabularSSL',
    'input_dim': processed_data.shape[1]
})

# Train model
model = instantiate(config)
model.train(processed_data)
```

## Model Configuration

### Custom Architecture

```python
# Create configuration with custom architecture
config = OmegaConf.create({
    '_target_': 'tabular_ssl.models.TabularSSL',
    'input_dim': 10,
    'sequence_encoder': {
        '_target_': 'tabular_ssl.models.encoders.TransformerEncoder',
        'input_dim': 10,
        'hidden_dim': 512,    # Larger hidden dimension
        'num_layers': 6,      # More transformer layers
        'num_heads': 8,       # More attention heads
        'dropout': 0.2        # Higher dropout
    },
    'mask_ratio': 0.2         # Higher masking ratio
})

# Initialize model
model = instantiate(config)
```

### Training Configuration

```python
# Create training configuration
train_config = OmegaConf.create({
    'batch_size': 64,         # Larger batch size
    'epochs': 200,            # More epochs
    'learning_rate': 5e-5,    # Lower learning rate
    'weight_decay': 1e-4      # L2 regularization
})

# Train model
history = model.train(
    data=processed_data,
    **train_config
)
```

## Model Evaluation

### Computing Metrics

```python
from tabular_ssl.utils import evaluate_model

metrics = evaluate_model(
    model,
    test_data,
    metrics=['accuracy', 'f1', 'precision', 'recall']
)
print(metrics)
```

### Visualization

```python
from tabular_ssl.utils import plot_training_history
import matplotlib.pyplot as plt

# Plot training history
fig = plot_training_history(history)
plt.show()

# Save the plot
fig.savefig('training_history.png')
```

## Model Persistence

### Saving and Loading

```python
# Save model
model.save('my_model.pt')

# Load model
loaded_model = TabularSSL.load('my_model.pt')
```

## Using Different Encoders

### Transformer Encoder

```python
config = OmegaConf.create({
    '_target_': 'tabular_ssl.models.TabularSSL',
    'input_dim': 10,
    'sequence_encoder': {
        '_target_': 'tabular_ssl.models.encoders.TransformerEncoder',
        'input_dim': 10,
        'hidden_dim': 256,
        'num_layers': 4,
        'num_heads': 4
    }
})
```

### RNN Encoder

```python
config = OmegaConf.create({
    '_target_': 'tabular_ssl.models.TabularSSL',
    'input_dim': 10,
    'sequence_encoder': {
        '_target_': 'tabular_ssl.models.encoders.RNNEncoder',
        'input_dim': 10,
        'hidden_dim': 256,
        'num_layers': 2,
        'bidirectional': true
    }
})
```

### LSTM Encoder

```python
config = OmegaConf.create({
    '_target_': 'tabular_ssl.models.TabularSSL',
    'input_dim': 10,
    'sequence_encoder': {
        '_target_': 'tabular_ssl.models.encoders.LSTMEncoder',
        'input_dim': 10,
        'hidden_dim': 256,
        'num_layers': 2,
        'bidirectional': true
    }
})
```

## Understanding Hydra Configuration

### Basic Concepts

Hydra is a framework for elegantly configuring complex applications. Here are the key concepts:

1. **Configuration Files**
   ```yaml
   # configs/model/default.yaml
   _target_: tabular_ssl.models.TabularSSL
   input_dim: 10
   sequence_encoder:
     _target_: tabular_ssl.models.encoders.TransformerEncoder
     input_dim: 10
     hidden_dim: 256
   ```

2. **Configuration Groups**
   ```yaml
   # configs/model/transformer.yaml
   _target_: tabular_ssl.models.encoders.TransformerEncoder
   input_dim: ${model.input_dim}
   hidden_dim: 256
   num_heads: 4

   # configs/model/rnn.yaml
   _target_: tabular_ssl.models.encoders.RNNEncoder
   input_dim: ${model.input_dim}
   hidden_dim: 256
   num_layers: 2
   ```

3. **Configuration Composition**
   ```yaml
   # configs/experiment/transformer_experiment.yaml
   defaults:
     - model: transformer
     - data: default
     - trainer: default
   ```

### Using Hydra in Code

1. **Loading Configurations**
   ```python
   from hydra import compose, initialize
   
   # Initialize Hydra
   with initialize(config_path="configs"):
       # Load default config
       config = compose(config_name="config")
       
       # Load specific experiment
       experiment_config = compose(config_name="experiment/transformer_experiment")
   ```

2. **Instantiating Objects**
   ```python
   from hydra.utils import instantiate
   
   # Create model from config
   model = instantiate(config.model)
   
   # Create optimizer
   optimizer = instantiate(config.optimizer, params=model.parameters())
   ```

3. **Overriding Configuration**
   ```python
   # Override specific values
   config = compose(
       config_name="config",
       overrides=["model.sequence_encoder.hidden_dim=512"]
   )
   ```

### Advanced Features

1. **Variable Interpolation**
   ```yaml
   # configs/model/default.yaml
   input_dim: 10
   sequence_encoder:
     input_dim: ${model.input_dim}  # References parent config
     hidden_dim: ${oc.env:HIDDEN_DIM,256}  # Uses environment variable with default
   ```

2. **Configuration Inheritance**
   ```yaml
   # configs/model/base.yaml
   _target_: tabular_ssl.models.TabularSSL
   input_dim: 10
   
   # configs/model/large.yaml
   defaults:
     - base
   hidden_dim: 512
   num_layers: 6
   ```

3. **Structured Configs**
   ```python
   from dataclasses import dataclass
   from omegaconf import MISSING
   
   @dataclass
   class ModelConfig:
       _target_: str = MISSING
       input_dim: int = MISSING
       hidden_dim: int = 256
   
   @dataclass
   class Config:
       model: ModelConfig = MISSING
   ```

### Best Practices

1. **Configuration Organization**
   - Keep related configs together
   - Use meaningful names
   - Document configuration options

2. **Default Values**
   - Provide sensible defaults
   - Use type hints
   - Document parameter ranges

3. **Error Handling**
   ```python
   from omegaconf import OmegaConf
   
   # Validate config
   try:
       OmegaConf.to_container(config, resolve=True)
   except Exception as e:
       print(f"Invalid configuration: {e}")
   ```

4. **Configuration Logging**
   ```python
   # Log configuration
   print(OmegaConf.to_yaml(config))
   
   # Save configuration
   OmegaConf.save(config, "config.yaml")
   ```

## Next Steps

- Explore [How-to Guides](../how-to-guides/index.md) for more specific use cases
- Check the [API Reference](../reference/api.md) for detailed documentation
- Learn about [SSL Methods](../explanation/ssl-methods.md) in depth 