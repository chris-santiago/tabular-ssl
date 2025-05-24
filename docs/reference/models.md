# Models Reference

This section provides detailed documentation of the model components and configurations available in Tabular SSL.

## Component Registry

The `ComponentRegistry` is a central registry that manages all model components.

```python
from tabular_ssl.models.base import ComponentRegistry
```

### Registry Methods

#### `register(name: str)`

Register a component class with the registry.

```python
@ComponentRegistry.register("custom_component")
class CustomComponent(BaseComponent):
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        # Implementation...
```

#### `get(name: str) -> Type[BaseComponent]`

Get a component class by name.

```python
component_cls = ComponentRegistry.get("transformer")
```

#### `list_components() -> Dict[str, Type[BaseComponent]]`

Get a dictionary of all registered components.

```python
available_components = ComponentRegistry.list_components()
```

## Base Components

### `BaseComponent`

The abstract base class for all model components.

```python
from tabular_ssl.models.base import BaseComponent
```

#### Constructor

```python
def __init__(self, config: ComponentConfig):
    super().__init__()
    self.config = config
    self._validate_config()
```

#### Methods

##### `_validate_config() -> None`

Validate the component configuration.

##### `forward(x: torch.Tensor) -> torch.Tensor`

Abstract method for the forward pass.

### `EventEncoder`

Base class for event encoders.

```python
from tabular_ssl.models.base import EventEncoder
```

### `SequenceEncoder`

Base class for sequence encoders.

```python
from tabular_ssl.models.base import SequenceEncoder
```

### `EmbeddingLayer`

Base class for embedding layers.

```python
from tabular_ssl.models.base import EmbeddingLayer
```

### `ProjectionHead`

Base class for projection heads.

```python
from tabular_ssl.models.base import ProjectionHead
```

### `PredictionHead`

Base class for prediction heads.

```python
from tabular_ssl.models.base import PredictionHead
```

## Component Configurations

### `ComponentConfig`

Base configuration for components.

```python
from tabular_ssl.models.base import ComponentConfig
```

#### Fields

- `name` (str): Name of the component
- `type` (str): Type of the component

### `MLPConfig`

Configuration for MLP-based components.

```python
from tabular_ssl.models.components import MLPConfig
```

#### Fields

- `input_dim` (int): Input dimension
- `hidden_dims` (List[int]): List of hidden dimensions
- `output_dim` (int): Output dimension
- `dropout` (float, default=0.1): Dropout rate
- `use_batch_norm` (bool, default=True): Whether to use batch normalization

### `SequenceModelConfig`

Base configuration for sequence models.

```python
from tabular_ssl.models.components import SequenceModelConfig
```

#### Fields

- `input_dim` (int): Input dimension
- `hidden_dim` (int): Hidden dimension
- `num_layers` (int, default=1): Number of layers
- `dropout` (float, default=0.1): Dropout rate
- `bidirectional` (bool, default=False): Whether to use bidirectional processing

### `TransformerConfig`

Configuration for Transformer models.

```python
from tabular_ssl.models.components import TransformerConfig
```

#### Fields

- All fields from `SequenceModelConfig`
- `num_heads` (int, default=4): Number of attention heads
- `dim_feedforward` (int): Dimension of feedforward network

### `S4Config`

Configuration for S4 models.

```python
from tabular_ssl.models.s4 import S4Config
```

#### Fields

- `d_model` (int): Model dimension
- `d_state` (int): State dimension
- `dropout` (float, default=0.1): Dropout rate
- `bidirectional` (bool, default=False): Whether to use bidirectional processing
- `max_sequence_length` (int, default=2048): Maximum sequence length

### `EmbeddingConfig`

Configuration for embedding layers.

```python
from tabular_ssl.models.components import EmbeddingConfig
```

#### Fields

- `embedding_dims` (List[tuple[int, int]]): List of (num_categories, embedding_dim) tuples
- `dropout` (float, default=0.1): Dropout rate

### `ProjectionHeadConfig`

Configuration for projection heads.

```python
from tabular_ssl.models.components import ProjectionHeadConfig
```

#### Fields

- `input_dim` (int): Input dimension
- `hidden_dims` (List[int]): List of hidden dimensions
- `output_dim` (int): Output dimension
- `dropout` (float, default=0.1): Dropout rate
- `use_batch_norm` (bool, default=True): Whether to use batch normalization

### `PredictionHeadConfig`

Configuration for prediction heads.

```python
from tabular_ssl.models.components import PredictionHeadConfig
```

#### Fields

- `input_dim` (int): Input dimension
- `num_classes` (int): Number of output classes
- `hidden_dims` (Optional[List[int]], default=None): Optional list of hidden dimensions
- `dropout` (float, default=0.1): Dropout rate
- `use_batch_norm` (bool, default=True): Whether to use batch normalization

## Component Implementations

### Event Encoders

#### `MLPEventEncoder`

A simple MLP-based event encoder.

```python
from tabular_ssl.models.components import MLPEventEncoder
```

#### `AutoEncoderEventEncoder`

Autoencoder-based event encoder.

```python
from tabular_ssl.models.components import AutoEncoderEventEncoder
```

#### Methods

##### `decode(z: torch.Tensor) -> torch.Tensor`

Decode the latent representation.

##### `reconstruction_loss(x: torch.Tensor) -> torch.Tensor`

Compute reconstruction loss.

#### `ContrastiveEventEncoder`

Contrastive learning-based event encoder.

```python
from tabular_ssl.models.components import ContrastiveEventEncoder
```

#### Methods

##### `contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor`

Compute contrastive loss between two views.

### Sequence Encoders

#### `RNNSequenceModel`

Basic RNN sequence model.

```python
from tabular_ssl.models.components import RNNSequenceModel
```

#### `LSTMSequenceModel`

LSTM sequence model.

```python
from tabular_ssl.models.components import LSTMSequenceModel
```

#### `GRUSequenceModel`

GRU sequence model.

```python
from tabular_ssl.models.components import GRUSequenceModel
```

#### `TransformerSequenceModel`

Transformer sequence model.

```python
from tabular_ssl.models.components import TransformerSequenceModel
```

#### `S4Model`

S4 sequence model with multiple S4 blocks.

```python
from tabular_ssl.models.s4 import S4Model
```

### Embedding Layers

#### `CategoricalEmbedding`

Embedding layer for categorical variables.

```python
from tabular_ssl.models.components import CategoricalEmbedding
```

### Projection Heads

#### `MLPProjectionHead`

MLP-based projection head.

```python
from tabular_ssl.models.components import MLPProjectionHead
```

### Prediction Heads

#### `ClassificationHead`

Classification head.

```python
from tabular_ssl.models.components import ClassificationHead
```

### Corruption Strategies

#### `RandomMasking`

Random masking corruption strategy.

```python
from tabular_ssl.models.components import RandomMasking
```

#### `GaussianNoise`

Gaussian noise corruption strategy.

```python
from tabular_ssl.models.components import GaussianNoise
```

#### `SwappingCorruption`

Feature swapping corruption strategy.

```python
from tabular_ssl.models.components import SwappingCorruption
```

#### `VIMECorruption`

VIME-style corruption strategy.

```python
from tabular_ssl.models.components import VIMECorruption
```

#### `CorruptionPipeline`

Pipeline of corruption strategies.

```python
from tabular_ssl.models.components import CorruptionPipeline
```

## Base Model

### `BaseModel`

Base model class for self-supervised sequence modeling.

```python
from tabular_ssl.models.base import BaseModel
```

#### Constructor

```python
def __init__(self, config: DictConfig):
    super().__init__()
    self.config = config
    
    # Convert Hydra configs to ComponentConfigs
    self.component_configs = {
        name: ComponentConfig.from_hydra(cfg)
        for name, cfg in config.model.items() if cfg is not None
    }
    
    # Initialize components
    self.components = {
        name: self._init_component(cfg)
        for name, cfg in self.component_configs.items()
    }
    
    self.save_hyperparameters(OmegaConf.to_container(config, resolve=True))
```

#### Methods

##### `_init_component(config: ComponentConfig) -> BaseComponent`

Initialize a component from its configuration.

##### `forward(x: torch.Tensor) -> torch.Tensor`

Forward pass through the model.

##### `training_step(batch, batch_idx)`

Training step (to be implemented by subclasses).

##### `validation_step(batch, batch_idx)`

Validation step (to be implemented by subclasses).

##### `configure_optimizers()`

Configure optimizers.

## Configuration Examples

### Event Encoder Configuration

```yaml
# Example configuration for an MLP event encoder
name: mlp_encoder
type: mlp_event_encoder
input_dim: 64
hidden_dims: [128, 256]
output_dim: 512
dropout: 0.1
use_batch_norm: true
```

### Sequence Encoder Configuration

```yaml
# Example configuration for a Transformer sequence encoder
name: transformer_encoder
type: transformer
input_dim: 512
hidden_dim: 512
num_layers: 4
num_heads: 8
dim_feedforward: 2048
dropout: 0.1
bidirectional: true
```

### Model Configuration

```yaml
# Example model configuration
model:
  event_encoder:
    name: mlp_event_encoder
    type: mlp_event_encoder
    input_dim: 64
    hidden_dims: [128, 256]
    output_dim: 512
    dropout: 0.1
    use_batch_norm: true
  sequence_encoder:
    name: s4
    type: s4
    input_dim: 512
    hidden_dim: 64
    num_layers: 2
    dropout: 0.1
    bidirectional: true
    max_sequence_length: 2048
  projection_head:
    name: mlp_projection
    type: mlp_projection
    input_dim: 512
    hidden_dims: [256]
    output_dim: 128
    dropout: 0.1
    use_batch_norm: true
optimizer:
  _target_: torch.optim.Adam
  lr: 0.001
``` 