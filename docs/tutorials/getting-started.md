# Getting Started with Tabular SSL

This tutorial will guide you through the process of setting up and using the Tabular SSL library for self-supervised learning on tabular data.

## Installation

First, install the library using pip:

```bash
pip install tabular-ssl
```

## Basic Usage

Here's a simple example of how to use Tabular SSL:

```python
import pandas as pd
from tabular_ssl import TabularSSL
from tabular_ssl.data import DataLoader

# Load your data
data_loader = DataLoader()
data = data_loader.load_data('your_data.csv')

# Initialize the model
model = TabularSSL(
    input_dim=data.shape[1],
    hidden_dim=256,
    num_layers=4,
    num_heads=4
)

# Train the model
history = model.train(
    data=data,
    batch_size=32,
    epochs=100,
    learning_rate=1e-4
)

# Make predictions
predictions = model.predict(new_data)
```

## Step-by-Step Guide

### 1. Data Preparation

First, prepare your data:

```python
# Load and preprocess your data
data_loader = DataLoader()
data = data_loader.load_data('your_data.csv')

# If you have categorical columns, specify them
categorical_cols = ['category1', 'category2']
data = data_loader.preprocess(data, categorical_cols=categorical_cols)
```

### 2. Model Configuration

Configure your model with appropriate parameters:

```python
model = TabularSSL(
    input_dim=data.shape[1],  # Number of features
    hidden_dim=256,           # Hidden layer dimension
    num_layers=4,             # Number of transformer layers
    num_heads=4,              # Number of attention heads
    dropout=0.1,              # Dropout rate
    mask_ratio=0.15           # Feature masking ratio
)
```

### 3. Training

Train the model using self-supervised learning:

```python
history = model.train(
    data=data,
    batch_size=32,
    epochs=100,
    learning_rate=1e-4
)
```

### 4. Evaluation

Evaluate your model's performance:

```python
from tabular_ssl.utils import evaluate_model, plot_training_history

# Plot training history
plot_training_history(history)

# Evaluate model performance
metrics = evaluate_model(model, test_data, metrics=['accuracy', 'f1'])
print(metrics)
```

## Next Steps

- Check out the [Basic Usage](basic-usage.md) tutorial for more advanced examples
- Explore the [API Reference](../reference/api.md) for detailed documentation
- Learn about different [SSL Methods](../explanation/ssl-methods.md) you can use

## Common Issues and Solutions

### Memory Issues

If you encounter memory issues with large datasets:

```python
# Use a smaller batch size
model.train(data, batch_size=16)

# Or reduce model complexity
model = TabularSSL(
    input_dim=data.shape[1],
    hidden_dim=128,  # Reduced from 256
    num_layers=2,    # Reduced from 4
    num_heads=2      # Reduced from 4
)
```

### Training Stability

For better training stability:

```python
# Use a lower learning rate
model.train(data, learning_rate=1e-5)

# Increase the number of epochs
model.train(data, epochs=200)
```

## Additional Resources

- [GitHub Repository](https://github.com/yourusername/tabular-ssl)
- [Issue Tracker](https://github.com/yourusername/tabular-ssl/issues)
- [Contributing Guide](../CONTRIBUTING.md) 