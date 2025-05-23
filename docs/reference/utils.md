# Utility Functions Reference

This section provides detailed documentation of the utility functions and tools in Tabular SSL.

## Evaluation Utilities

### Model Evaluation

```python
from tabular_ssl.utils import evaluate_model

metrics = evaluate_model(
    model,
    test_data,
    metrics=['accuracy', 'f1', 'precision', 'recall']
)
```

### Cross-Validation

```python
from tabular_ssl.utils import cross_validate

cv_results = cross_validate(
    model,
    data,
    n_splits=5,
    metrics=['accuracy', 'f1']
)
```

## Visualization Utilities

### Training History

```python
from tabular_ssl.utils import plot_training_history

fig = plot_training_history(history)
fig.show()
```

### Performance Plots

```python
from tabular_ssl.utils import plot_performance

fig = plot_performance(
    model,
    test_data,
    plot_types=['confusion_matrix', 'roc_curve']
)
fig.show()
```

## Model Interpretation

### Feature Importance

```python
from tabular_ssl.utils import get_feature_importance

importance = get_feature_importance(model, test_data)
```

### SHAP Values

```python
from tabular_ssl.utils import get_shap_values

shap_values = get_shap_values(model, test_data)
```

## Hyperparameter Tuning

### Grid Search

```python
from tabular_ssl.utils import grid_search

best_params = grid_search(
    model_class=TabularSSL,
    param_grid={
        'hidden_dim': [128, 256, 512],
        'num_layers': [2, 4, 6]
    },
    train_data=train_data,
    val_data=val_data
)
```

### Random Search

```python
from tabular_ssl.utils import random_search

best_params = random_search(
    model_class=TabularSSL,
    param_distributions={
        'hidden_dim': [128, 256, 512],
        'num_layers': [2, 4, 6]
    },
    train_data=train_data,
    val_data=val_data,
    n_iter=10
)
```

## Data Utilities

### Data Validation

```python
from tabular_ssl.utils import validate_data

validation_results = validate_data(data)
```

### Feature Selection

```python
from tabular_ssl.utils import select_features

selected_features = select_features(
    data,
    target_col='target',
    method='importance',
    threshold=0.01
)
```

## Model Utilities

### Model Saving

```python
from tabular_ssl.utils import save_model

save_model(model, 'model.pt')
```

### Model Loading

```python
from tabular_ssl.utils import load_model

model = load_model('model.pt')
```

## Training Utilities

### Learning Rate Scheduling

```python
from tabular_ssl.utils import get_lr_scheduler

scheduler = get_lr_scheduler(
    initial_lr=1e-3,
    scheduler_type='cosine',
    warmup_epochs=5
)
```

### Early Stopping

```python
from tabular_ssl.utils import EarlyStopping

early_stopping = EarlyStopping(
    patience=10,
    min_delta=0.001
)
```

## Common Functions

### Metrics

```python
from tabular_ssl.utils import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

# Compute metrics
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
```

### Data Processing

```python
from tabular_ssl.utils import (
    normalize_data,
    encode_categorical,
    handle_missing
)

# Process data
normalized_data = normalize_data(data)
encoded_data = encode_categorical(data, categorical_cols)
cleaned_data = handle_missing(data, strategy='mean')
```

## Best Practices

1. Use appropriate evaluation metrics
2. Implement proper cross-validation
3. Visualize results for better understanding
4. Document utility function usage
5. Handle errors gracefully
6. Use type hints for better code clarity
7. Add proper docstrings
8. Include examples in documentation

## Related Resources

- [API Reference](api.md) - Complete API documentation
- [How-to Guides](../how-to-guides/evaluation.md) - Evaluation guides
- [Tutorials](../tutorials/basic-usage.md) - Usage examples 