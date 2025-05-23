# Model Training Guide

This guide covers best practices for training models with Tabular SSL.

## Basic Training

### Simple Training Loop

```python
from tabular_ssl import TabularSSL

# Initialize model
model = TabularSSL(input_dim=10)

# Train model
history = model.train(
    data=train_data,
    batch_size=32,
    epochs=100
)
```

### Training with Validation

```python
# Split data into train and validation sets
from sklearn.model_selection import train_test_split

train_data, val_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

# Train with validation
history = model.train(
    data=train_data,
    validation_data=val_data,
    batch_size=32,
    epochs=100
)
```

## Advanced Training

### Custom Training Configuration

```python
history = model.train(
    data=train_data,
    batch_size=64,
    epochs=200,
    learning_rate=1e-4,
    weight_decay=1e-5,
    early_stopping=True,
    patience=10,
    min_delta=0.001
)
```

### Learning Rate Scheduling

```python
from tabular_ssl.utils import get_lr_scheduler

# Create learning rate scheduler
lr_scheduler = get_lr_scheduler(
    initial_lr=1e-3,
    scheduler_type='cosine',
    warmup_epochs=5
)

# Train with scheduler
history = model.train(
    data=train_data,
    learning_rate=1e-3,
    lr_scheduler=lr_scheduler
)
```

## Monitoring Training

### Logging Metrics

```python
from tabular_ssl.utils import TrainingLogger

# Initialize logger
logger = TrainingLogger(log_dir='logs')

# Train with logging
history = model.train(
    data=train_data,
    logger=logger,
    log_interval=100
)
```

### Visualizing Progress

```python
from tabular_ssl.utils import plot_training_history

# Plot training metrics
fig = plot_training_history(history)
fig.show()
```

## Model Checkpointing

### Saving Checkpoints

```python
# Save model checkpoints
history = model.train(
    data=train_data,
    save_dir='checkpoints',
    save_best_only=True,
    save_frequency=5
)
```

### Loading Checkpoints

```python
# Load best model
model = TabularSSL.load('checkpoints/best_model.pt')

# Load specific checkpoint
model = TabularSSL.load('checkpoints/model_epoch_50.pt')
```

## Hyperparameter Tuning

### Grid Search

```python
from tabular_ssl.utils import grid_search

# Define parameter grid
param_grid = {
    'hidden_dim': [128, 256, 512],
    'num_layers': [2, 4, 6],
    'dropout': [0.1, 0.2, 0.3]
}

# Perform grid search
best_params = grid_search(
    model_class=TabularSSL,
    param_grid=param_grid,
    train_data=train_data,
    val_data=val_data
)
```

### Random Search

```python
from tabular_ssl.utils import random_search

# Define parameter distributions
param_distributions = {
    'hidden_dim': [128, 256, 512],
    'num_layers': [2, 4, 6],
    'dropout': [0.1, 0.2, 0.3]
}

# Perform random search
best_params = random_search(
    model_class=TabularSSL,
    param_distributions=param_distributions,
    train_data=train_data,
    val_data=val_data,
    n_iter=10
)
```

## Best Practices

1. Always use validation data during training
2. Implement early stopping to prevent overfitting
3. Use learning rate scheduling for better convergence
4. Monitor training metrics regularly
5. Save model checkpoints
6. Experiment with different hyperparameters
7. Use appropriate batch sizes for your data
8. Normalize your data before training

## Common Issues and Solutions

### Overfitting

```python
# Increase dropout
model = TabularSSL(
    input_dim=10,
    dropout=0.3  # Increased from default
)

# Add regularization
history = model.train(
    data=train_data,
    weight_decay=1e-4
)
```

### Underfitting

```python
# Increase model capacity
model = TabularSSL(
    input_dim=10,
    hidden_dim=512,  # Increased from default
    num_layers=6     # Increased from default
)

# Train for more epochs
history = model.train(
    data=train_data,
    epochs=300  # Increased from default
)
```

## Related Resources

- [Data Preparation](data-preparation.md) - Preparing your data for training
- [Evaluation](evaluation.md) - Evaluating your trained model
- [API Reference](../reference/api.md) - Detailed API documentation 