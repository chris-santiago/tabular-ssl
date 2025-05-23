# Model Evaluation Guide

This guide covers how to evaluate and interpret your Tabular SSL models.

## Basic Evaluation

### Computing Metrics

```python
from tabular_ssl.utils import evaluate_model

# Evaluate model performance
metrics = evaluate_model(
    model,
    test_data,
    metrics=['accuracy', 'f1', 'precision', 'recall']
)
print(metrics)
```

### Cross-Validation

```python
from tabular_ssl.utils import cross_validate

# Perform k-fold cross-validation
cv_results = cross_validate(
    model,
    data,
    n_splits=5,
    metrics=['accuracy', 'f1']
)
print(cv_results)
```

## Advanced Evaluation

### Custom Metrics

```python
from tabular_ssl.utils import CustomMetric

# Define custom metric
def custom_metric(y_true, y_pred):
    # Your custom metric implementation
    return score

# Evaluate with custom metric
metrics = evaluate_model(
    model,
    test_data,
    metrics=['accuracy', CustomMetric(custom_metric)]
)
```

### Model Comparison

```python
from tabular_ssl.utils import compare_models

# Compare multiple models
comparison = compare_models(
    models=[model1, model2, model3],
    test_data=test_data,
    metrics=['accuracy', 'f1']
)
print(comparison)
```

## Visualization

### Training History

```python
from tabular_ssl.utils import plot_training_history

# Plot training metrics
fig = plot_training_history(history)
fig.show()
```

### Performance Plots

```python
from tabular_ssl.utils import plot_performance

# Plot various performance metrics
fig = plot_performance(
    model,
    test_data,
    plot_types=['confusion_matrix', 'roc_curve', 'precision_recall']
)
fig.show()
```

## Model Interpretation

### Feature Importance

```python
from tabular_ssl.utils import get_feature_importance

# Get feature importance scores
importance = get_feature_importance(model, test_data)
print(importance)
```

### SHAP Values

```python
from tabular_ssl.utils import get_shap_values

# Compute SHAP values
shap_values = get_shap_values(model, test_data)

# Plot SHAP summary
plot_shap_summary(shap_values, test_data)
```

## Error Analysis

### Error Distribution

```python
from tabular_ssl.utils import analyze_errors

# Analyze prediction errors
error_analysis = analyze_errors(
    model,
    test_data,
    analysis_types=['distribution', 'correlation']
)
print(error_analysis)
```

### Error Visualization

```python
from tabular_ssl.utils import plot_errors

# Plot error analysis
fig = plot_errors(
    model,
    test_data,
    plot_types=['residuals', 'error_distribution']
)
fig.show()
```

## Best Practices

1. Use multiple evaluation metrics
2. Perform cross-validation for robust results
3. Compare against baseline models
4. Analyze error patterns
5. Visualize results for better understanding
6. Consider domain-specific metrics
7. Document evaluation methodology
8. Validate results with statistical tests

## Common Issues and Solutions

### Unbalanced Data

```python
from tabular_ssl.utils import balanced_metrics

# Use balanced metrics
metrics = evaluate_model(
    model,
    test_data,
    metrics=['balanced_accuracy', 'f1']
)
```

### Small Test Sets

```python
# Use bootstrapping for small test sets
from tabular_ssl.utils import bootstrap_evaluation

results = bootstrap_evaluation(
    model,
    test_data,
    n_bootstrap=1000,
    metrics=['accuracy', 'f1']
)
```

## Related Resources

- [Model Training](model-training.md) - Training your model
- [Data Preparation](data-preparation.md) - Preparing your data
- [API Reference](../reference/api.md) - Detailed API documentation 