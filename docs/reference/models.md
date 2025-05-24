# Models Reference

This section provides detailed documentation of the model components and architecture available in Tabular SSL.

## Architecture Overview

Tabular SSL uses a simplified modular architecture where components are directly instantiated using Hydra's `_target_` mechanism. This approach is cleaner and more straightforward than the previous registry-based system.

## Base Components

### `BaseModel`

The main model class that orchestrates all components.

```python
from tabular_ssl.models.base import BaseModel
```

#### Constructor

```python
def __init__(
    self,
    event_encoder,
    sequence_encoder=None,
    embedding=None,
    projection_head=None,
    prediction_head=None,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    optimizer_type: str = "adamw",
    scheduler_type: str = "cosine"
):
```

#### Key Methods

- `forward(x)`: Main forward pass through all components
- `training_step(batch, batch_idx)`: PyTorch Lightning training step
- `validation_step(batch, batch_idx)`: PyTorch Lightning validation step
- `configure_optimizers()`: Optimizer and scheduler configuration

### `EventEncoder`

Base class for components that encode individual events or timesteps.

```python
from tabular_ssl.models.base import EventEncoder
```

All event encoders should inherit from this class and implement the `forward` method:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: Input tensor of shape (batch_size, seq_len, input_dim)
    Returns:
        Encoded tensor of shape (batch_size, seq_len, output_dim)
    """
```

### `SequenceEncoder`

Base class for components that encode sequences of events.

```python
from tabular_ssl.models.base import SequenceEncoder
```

Sequence encoders process the output of event encoders:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: Input tensor of shape (batch_size, seq_len, input_dim)
    Returns:
        Encoded sequence of shape (batch_size, seq_len, output_dim)
    """
```

### `EmbeddingLayer`

Base class for embedding layers that handle categorical features.

```python
from tabular_ssl.models.base import EmbeddingLayer
```

### `ProjectionHead`

Base class for projection heads that transform representations.

```python
from tabular_ssl.models.base import ProjectionHead
```

### `PredictionHead`

Base class for prediction heads that generate final outputs.

```python
from tabular_ssl.models.base import PredictionHead
```

## Available Components

### Event Encoders

#### `MLPEventEncoder`

Multi-layer perceptron for encoding individual events.

```python
from tabular_ssl.models.components import MLPEventEncoder
```

**Constructor Parameters:**
- `input_dim` (int): Input feature dimension
- `hidden_dims` (List[int]): List of hidden layer dimensions
- `output_dim` (int): Output dimension
- `dropout` (float): Dropout rate (default: 0.1)
- `activation` (str): Activation function ("relu", "gelu", "leaky_relu")
- `use_batch_norm` (bool): Whether to use batch normalization

**Example Configuration:**
```yaml
_target_: tabular_ssl.models.components.MLPEventEncoder
input_dim: 64
hidden_dims: [128, 256]
output_dim: 512
dropout: 0.1
activation: relu
use_batch_norm: true
```

### Sequence Encoders

#### `TransformerSequenceEncoder`

Transformer-based sequence encoder using self-attention.

```python
from tabular_ssl.models.components import TransformerSequenceEncoder
```

**Constructor Parameters:**
- `input_dim` (int): Input dimension
- `hidden_dim` (int): Hidden dimension
- `num_layers` (int): Number of transformer layers
- `num_heads` (int): Number of attention heads
- `dim_feedforward` (int): Feedforward network dimension
- `dropout` (float): Dropout rate
- `max_seq_length` (int): Maximum sequence length for positional encoding

**Example Configuration:**
```yaml
_target_: tabular_ssl.models.components.TransformerSequenceEncoder
input_dim: 512
hidden_dim: 512
num_layers: 4
num_heads: 8
dim_feedforward: 2048
dropout: 0.1
max_seq_length: 2048
```

#### `S4SequenceEncoder`

Structured State Space (S4) model for efficient long sequence processing.

```python
from tabular_ssl.models.components import S4SequenceEncoder
```

**Constructor Parameters:**
- `input_dim` (int): Input dimension
- `hidden_dim` (int): Hidden state dimension
- `num_layers` (int): Number of S4 layers
- `dropout` (float): Dropout rate
- `bidirectional` (bool): Whether to use bidirectional processing
- `max_sequence_length` (int): Maximum sequence length

**Example Configuration:**
```yaml
_target_: tabular_ssl.models.components.S4SequenceEncoder
input_dim: 512
hidden_dim: 64
num_layers: 2
dropout: 0.1
bidirectional: true
max_sequence_length: 2048
```

#### `RNNSequenceEncoder`

RNN-based sequence encoder (LSTM, GRU, or vanilla RNN).

```python
from tabular_ssl.models.components import RNNSequenceEncoder
```

**Constructor Parameters:**
- `input_dim` (int): Input dimension
- `hidden_dim` (int): Hidden dimension
- `num_layers` (int): Number of RNN layers
- `rnn_type` (str): Type of RNN ("lstm", "gru", "rnn")
- `dropout` (float): Dropout rate
- `bidirectional` (bool): Whether to use bidirectional RNN

**Example Configuration:**
```yaml
_target_: tabular_ssl.models.components.RNNSequenceEncoder
input_dim: 128
hidden_dim: 128
num_layers: 2
rnn_type: lstm
dropout: 0.1
bidirectional: false
```

### Embedding Layers

#### `CategoricalEmbedding`

Handles embedding of categorical features with flexible dimensions.

```python
from tabular_ssl.models.components import CategoricalEmbedding
```

**Constructor Parameters:**
- `categorical_features` (List[Dict]): List of categorical feature specifications
- `default_embedding_dim` (int): Default embedding dimension
- `categorical_embedding_dims` (Dict[str, int]): Custom dimensions per feature

**Example Configuration:**
```yaml
_target_: tabular_ssl.models.components.CategoricalEmbedding
categorical_features:
  - name: category_1
    num_categories: 10
    embedding_dim: 32
  - name: category_2
    num_categories: 100
    embedding_dim: 64
```

### Projection Heads

#### `MLPProjectionHead`

MLP-based projection head for transforming representations.

```python
from tabular_ssl.models.components import MLPProjectionHead
```

**Constructor Parameters:**
- `input_dim` (int): Input dimension
- `hidden_dims` (List[int]): Hidden layer dimensions
- `output_dim` (int): Output dimension
- `dropout` (float): Dropout rate
- `activation` (str): Activation function

### Prediction Heads

#### `ClassificationHead`

Classification head for supervised learning tasks.

```python
from tabular_ssl.models.components import ClassificationHead
```

**Constructor Parameters:**
- `input_dim` (int): Input dimension
- `num_classes` (int): Number of output classes
- `hidden_dims` (List[int]): Optional hidden layers
- `dropout` (float): Dropout rate

## Utility Functions

### `create_mlp`

Utility function for creating MLP layers.

```python
from tabular_ssl.models.base import create_mlp

mlp = create_mlp(
    input_dim=64,
    hidden_dims=[128, 256],
    output_dim=512,
    dropout=0.1,
    activation="relu",
    use_batch_norm=True
)
```

## Component Instantiation

Components are instantiated using Hydra's `_target_` mechanism:

```python
# Direct instantiation
encoder = hydra.utils.instantiate({
    "_target_": "tabular_ssl.models.components.MLPEventEncoder",
    "input_dim": 64,
    "hidden_dims": [128, 256],
    "output_dim": 512
})

# From configuration
encoder = hydra.utils.instantiate(config.model.event_encoder)
```

## Model Assembly

The `BaseModel` class assembles all components:

```yaml
# configs/model/default.yaml
defaults:
  - event_encoder: mlp
  - sequence_encoder: transformer
  - projection_head: mlp
  - prediction_head: classification

_target_: tabular_ssl.models.base.BaseModel
learning_rate: 1.0e-4
weight_decay: 0.01
optimizer_type: adamw
scheduler_type: cosine
```

## Creating Custom Components

To create custom components, inherit from the appropriate base class:

```python
from tabular_ssl.models.base import EventEncoder

class CustomEventEncoder(EventEncoder):
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.reshape(-1, x.size(-1))).reshape(x.shape[:-1] + (-1,))
```

Then create a configuration file:

```yaml
# configs/model/event_encoder/custom.yaml
_target_: path.to.your.CustomEventEncoder
input_dim: 64
output_dim: 128
```

## Best Practices

1. **Component Compatibility**: Ensure output dimensions of one component match input dimensions of the next
2. **Memory Management**: Use appropriate batch sizes and sequence lengths for your hardware
3. **Hyperparameter Tuning**: Start with provided experiment configurations and adjust as needed
4. **Testing**: Test custom components with different input shapes before integration

## Common Patterns

### MLP-Only Model
```yaml
defaults:
  - event_encoder: mlp
  - sequence_encoder: null  # No sequence processing
```

### Transformer Model
```yaml
defaults:
  - event_encoder: mlp
  - sequence_encoder: transformer
```

### Long Sequence Model
```yaml
defaults:
  - event_encoder: mlp
  - sequence_encoder: s4  # Efficient for long sequences
```

## Troubleshooting

### Dimension Mismatches
Check that component dimensions are compatible:
```python
assert event_encoder.output_dim == sequence_encoder.input_dim
```

### Memory Issues
- Reduce batch size or sequence length
- Use gradient accumulation
- Enable mixed precision training

### Training Instability
- Lower learning rate
- Add gradient clipping
- Use layer normalization or batch normalization 