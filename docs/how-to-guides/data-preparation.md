# Data Preparation Guide

This guide covers best practices for preparing your data for use with Tabular SSL.

## Loading Data

### From CSV Files

```python
from tabular_ssl.data import DataLoader

# Initialize the data loader
data_loader = DataLoader()

# Load data from CSV
data = data_loader.load_data('path/to/your/data.csv')
```

### From Pandas DataFrame

```python
import pandas as pd

# Create or load your DataFrame
df = pd.DataFrame({
    'numeric_col': [1, 2, 3],
    'categorical_col': ['A', 'B', 'A']
})

# Use the data loader
data_loader = DataLoader()
processed_data = data_loader.preprocess(df)
```

## Handling Different Data Types

### Categorical Variables

```python
# Specify categorical columns
categorical_cols = ['category1', 'category2']
processed_data = data_loader.preprocess(
    data,
    categorical_cols=categorical_cols
)
```

### Numerical Variables

```python
# Numerical columns are automatically detected
# You can specify scaling options
processed_data = data_loader.preprocess(
    data,
    scale_numerical=True,  # Enable scaling
    scaler='standard'      # Use standard scaler
)
```

## Dealing with Missing Values

### Automatic Handling

```python
# The data loader automatically handles missing values
processed_data = data_loader.preprocess(
    data,
    handle_missing=True,  # Enable missing value handling
    missing_strategy='mean'  # Use mean imputation
)
```

### Manual Handling

```python
import pandas as pd
import numpy as np

# Fill missing values
data = data.fillna({
    'numeric_col': data['numeric_col'].mean(),
    'categorical_col': data['categorical_col'].mode()[0]
})
```

## Feature Engineering

### Creating New Features

```python
# Add interaction terms
data['interaction'] = data['feature1'] * data['feature2']

# Add polynomial features
data['feature1_squared'] = data['feature1'] ** 2
```

### Feature Selection

```python
from tabular_ssl.utils import select_features

# Select features based on importance
selected_features = select_features(
    data,
    target_col='target',
    method='importance',
    threshold=0.01
)
```

## Data Validation

### Checking Data Quality

```python
from tabular_ssl.utils import validate_data

# Validate data before processing
validation_results = validate_data(data)
print(validation_results)
```

### Common Issues and Solutions

1. **Inconsistent Data Types**
   ```python
   # Convert columns to correct types
   data['numeric_col'] = pd.to_numeric(data['numeric_col'])
   data['categorical_col'] = data['categorical_col'].astype('category')
   ```

2. **Outliers**
   ```python
   # Remove outliers
   data = data[data['numeric_col'].between(
       data['numeric_col'].quantile(0.01),
       data['numeric_col'].quantile(0.99)
   )]
   ```

## Best Practices

1. Always validate your data before processing
2. Handle missing values appropriately for your use case
3. Scale numerical features when necessary
4. Encode categorical variables properly
5. Check for and handle outliers
6. Document your preprocessing steps

## Related Resources

- [Model Training](model-training.md) - Next steps after data preparation
- [API Reference](../reference/api.md) - Detailed API documentation
- [Tutorials](../tutorials/index.md) - Step-by-step guides 