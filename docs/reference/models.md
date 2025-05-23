# Models Reference

This section provides detailed documentation of the model architectures and configurations available in Tabular SSL.

## TabularSSL

The main model class for self-supervised learning on tabular data.

```python
from tabular_ssl import TabularSSL
```

### Architecture

The TabularSSL model consists of:

1. **Input Layer**
   - Handles mixed data types (numerical and categorical)
   - Feature embedding layer
   - Positional encoding

2. **Transformer Encoder**
   - Multi-head self-attention
   - Feed-forward networks
   - Layer normalization
   - Residual connections

3. **Output Layer**
   - Feature reconstruction
   - Task-specific heads

### Configuration

The model uses Hydra's configuration system with the `_target_` pattern for flexible component instantiation.

#### Example Configuration

```yaml
_target_: tabular_ssl.models.TabularSSL

# Model components
event_encoder:
  _target_: tabular_ssl.models.encoders.EventEncoder
  input_dim: 10
  hidden_dims: [64, 32]
  output_dim: 16

sequence_encoder:
  _target_: tabular_ssl.models.encoders.TransformerEncoder
  input_dim: 16
  hidden_dim: 32
  num_layers: 2
  dropout: 0.1
  num_heads: 4

embedding:
  _target_: tabular_ssl.models.embeddings.FeatureEmbedding
  embedding_dims:
    - [5, 8]  # 5 categories, 8-dimensional embedding
    - [3, 4]  # 3 categories, 4-dimensional embedding

projection_head:
  _target_: tabular_ssl.models.heads.ProjectionHead
  input_dim: 32
  hidden_dim: 16
  output_dim: 8

prediction_head:
  _target_: tabular_ssl.models.heads.PredictionHead
  input_dim: 8
  num_classes: 2
  dropout: 0.1
```

#### Available Encoders

1. **Transformer Encoder**
   ```yaml
   _target_: tabular_ssl.models.encoders.TransformerEncoder
   input_dim: 16
   hidden_dim: 32
   num_layers: 2
   num_heads: 4
   dropout: 0.1
   ```

2. **RNN Encoder**
   ```yaml
   _target_: tabular_ssl.models.encoders.RNNEncoder
   input_dim: 16
   hidden_dim: 32
   num_layers: 2
   dropout: 0.1
   bidirectional: true
   ```

3. **LSTM Encoder**
   ```yaml
   _target_: tabular_ssl.models.encoders.LSTMEncoder
   input_dim: 16
   hidden_dim: 32
   num_layers: 2
   dropout: 0.1
   bidirectional: true
   ```

4. **GRU Encoder**
   ```yaml
   _target_: tabular_ssl.models.encoders.GRUEncoder
   input_dim: 16
   hidden_dim: 32
   num_layers: 2
   dropout: 0.1
   bidirectional: true
   ```

5. **SSM Encoder**
   ```yaml
   _target_: tabular_ssl.models.encoders.SSMEncoder
   input_dim: 16
   hidden_dim: 32
   state_dim: 32
   use_gate: true
   ```

6. **S4 Encoder**
   ```yaml
   _target_: tabular_ssl.models.encoders.S4Encoder
   input_dim: 16
   hidden_dim: 32
   max_sequence_length: 1024
   use_learnable_dt: true
   use_initial_state: true
   ```

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

##### `save(path)`

Save the model to disk.

**Parameters:**
- `path` (str): Path to save the model

##### `load(path)`

Load a model from disk.

**Parameters:**
- `path` (str): Path to the saved model

**Returns:**
- `TabularSSL`: Loaded model

## Model Components

### Feature Embedding

```python
from tabular_ssl.models.embeddings import FeatureEmbedding

embedding = FeatureEmbedding(
    input_dim=10,
    embedding_dim=64
)
```

### Transformer Encoder

```python
from tabular_ssl.models.encoders import TransformerEncoder

encoder = TransformerEncoder(
    input_dim=64,
    hidden_dim=256,
    num_layers=4,
    num_heads=4
)
```

### Task Heads

```python
from tabular_ssl.models.heads import TaskHead

head = TaskHead(
    input_dim=256,
    output_dim=10
)
```

## Model Variants

### TabularSSL-Large

```yaml
_target_: tabular_ssl.models.TabularSSLLarge
input_dim: 10
hidden_dim: 512
num_layers: 6
num_heads: 8
```

### TabularSSL-Small

```yaml
_target_: tabular_ssl.models.TabularSSLSmall
input_dim: 10
hidden_dim: 128
num_layers: 2
num_heads: 2
```

## Related Resources

- [API Reference](api.md) - Complete API documentation
- [How-to Guides](../how-to-guides/model-training.md) - Training guides
- [Explanation](../explanation/architecture.md) - Architecture details 