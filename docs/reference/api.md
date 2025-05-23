# API Reference

This section provides detailed technical documentation for all public APIs in the Tabular SSL library.

## Models

### TabularSSL

The main class for self-supervised learning on tabular data.

```python
from tabular_ssl import TabularSSL
```

#### Parameters

- `input_dim` (int): Dimension of input features
- `hidden_dim` (int, optional): Dimension of hidden layers. Defaults to 256.
- `num_layers` (int, optional): Number of transformer layers. Defaults to 4.
- `num_heads` (int, optional): Number of attention heads. Defaults to 4.
- `dropout` (float, optional): Dropout rate. Defaults to 0.1.
- `mask_ratio` (float, optional): Ratio of features to mask during training. Defaults to 0.15.

#### Methods

##### `train(data, batch_size=32, epochs=100, learning_rate=1e-4)`

Train the model using self-supervised learning.

**Parameters:**
- `data` (pd.DataFrame): Input data
- `batch_size` (int): Batch size for training
- `epochs` (int): Number of training epochs
- `learning_rate` (float): Learning rate

**Returns:**
- `dict`: Training history

##### `predict(data)`

Make predictions on new data.

**Parameters:**
- `data` (pd.DataFrame): Input data

**Returns:**
- `np.ndarray`: Model predictions

## Data Utilities

### DataLoader

Utility class for loading and preprocessing tabular data.

```python
from tabular_ssl.data import DataLoader
```

#### Methods

##### `load_data(file_path, target_col=None)`

Load data from a file.

**Parameters:**
- `file_path` (str): Path to the data file
- `target_col` (str, optional): Name of the target column

**Returns:**
- `pd.DataFrame`: Loaded data

##### `preprocess(data, categorical_cols=None)`

Preprocess the data.

**Parameters:**
- `data` (pd.DataFrame): Input data
- `categorical_cols` (list, optional): List of categorical column names

**Returns:**
- `pd.DataFrame`: Preprocessed data

## Utility Functions

### Evaluation

```python
from tabular_ssl.utils import evaluate_model
```

#### `evaluate_model(model, test_data, metrics=['accuracy', 'f1'])`

Evaluate model performance.

**Parameters:**
- `model`: Trained model
- `test_data` (pd.DataFrame): Test data
- `metrics` (list): List of metrics to compute

**Returns:**
- `dict`: Dictionary of metric scores

### Visualization

```python
from tabular_ssl.utils import plot_training_history
```

#### `plot_training_history(history)`

Plot training history.

**Parameters:**
- `history` (dict): Training history dictionary

**Returns:**
- `matplotlib.figure.Figure`: Plot figure 