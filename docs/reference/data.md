# Data Utilities Reference

This section provides detailed documentation of the data loading and preprocessing utilities in Tabular SSL.

## DataLoader

The main class for loading and preprocessing tabular data.

```python
from tabular_ssl.data import DataLoader
```

### Methods

#### `load_data(file_path, target_col=None)`

Load data from a file.

**Parameters:**
- `file_path` (str): Path to the data file
- `target_col` (str, optional): Name of the target column

**Returns:**
- `pd.DataFrame`: Loaded data

#### `preprocess(data, categorical_cols=None, scale_numerical=True, handle_missing=True)`

Preprocess the data.

**Parameters:**
- `data` (pd.DataFrame): Input data
- `categorical_cols` (list, optional): List of categorical column names
- `scale_numerical` (bool, optional): Whether to scale numerical features
- `handle_missing` (bool, optional): Whether to handle missing values

**Returns:**
- `pd.DataFrame`: Preprocessed data

## Data Transformers

### CategoricalTransformer

```python
from tabular_ssl.data import CategoricalTransformer

transformer = CategoricalTransformer(
    columns=['category1', 'category2'],
    encoding='onehot'
)
```

### NumericalTransformer

```python
from tabular_ssl.data import NumericalTransformer

transformer = NumericalTransformer(
    columns=['numeric1', 'numeric2'],
    scaling='standard'
)
```

## Data Validation

### DataValidator

```python
from tabular_ssl.data import DataValidator

validator = DataValidator(
    required_columns=['col1', 'col2'],
    data_types={
        'col1': 'numeric',
        'col2': 'categorical'
    }
)
```

## Data Splitting

### DataSplitter

```python
from tabular_ssl.data import DataSplitter

splitter = DataSplitter(
    test_size=0.2,
    val_size=0.1,
    random_state=42
)
```

## Feature Engineering

### FeatureEngineer

```python
from tabular_ssl.data import FeatureEngineer

engineer = FeatureEngineer(
    interactions=True,
    polynomials=True,
    degree=2
)
```

## Data Augmentation

### DataAugmenter

```python
from tabular_ssl.data import DataAugmenter

augmenter = DataAugmenter(
    noise_level=0.1,
    mask_ratio=0.15
)
```

## Common Operations

### Loading Data

```python
# Load from CSV
data = DataLoader().load_data('data.csv')

# Load from DataFrame
data = DataLoader().load_data(df)
```

### Preprocessing

```python
# Basic preprocessing
processed_data = DataLoader().preprocess(
    data,
    categorical_cols=['category1', 'category2']
)

# Advanced preprocessing
processed_data = DataLoader().preprocess(
    data,
    categorical_cols=['category1', 'category2'],
    scale_numerical=True,
    handle_missing=True,
    missing_strategy='mean'
)
```

### Data Splitting

```python
# Split data
train_data, val_data, test_data = DataSplitter().split(data)
```

### Feature Engineering

```python
# Create new features
engineered_data = FeatureEngineer().transform(data)
```

## Best Practices

1. Always validate data before processing
2. Handle missing values appropriately
3. Scale numerical features
4. Encode categorical variables
5. Split data before preprocessing
6. Document preprocessing steps
7. Save preprocessed data
8. Use appropriate data types

## Related Resources

- [API Reference](api.md) - Complete API documentation
- [How-to Guides](../how-to-guides/data-preparation.md) - Data preparation guides
- [Tutorials](../tutorials/getting-started.md) - Getting started guides 