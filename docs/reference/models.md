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
    self.event_encoder_config = ComponentConfig.from_hydra(config.model.event_encoder)
    self.sequence_encoder_config = ComponentConfig.from_hydra(config.model.sequence_encoder) if config.model.sequence_encoder else None
    self.embedding_config = ComponentConfig.from_hydra(config.model.embedding) if config.model.embedding else None
    self.projection_head_config = ComponentConfig.from_hydra(config.model.projection_head) if config.model.projection_head else None
    self.prediction_head_config = ComponentConfig.from_hydra(config.model.prediction_head) if config.model.prediction_head else None
    
    # Initialize components
    self.event_encoder = self._init_component(self.event_encoder_config)
    self.sequence_encoder = self._init_component(self.sequence_encoder_config) if self.sequence_encoder_config else None
    self.embedding_layer = self._init_component(self.embedding_config) if self.embedding_config else None
    self.projection_head = self._init_component(self.projection_head_config) if self.projection_head_config else None
    self.prediction_head = self._init_component(self.prediction_head_config) if self.prediction_head_config else None
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
# configs/model/event_encoder/mlp.yaml
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
# configs/model/sequence_encoder/transformer.yaml
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
# configs/model/default.yaml
defaults:
  - _self_
  - event_encoder: mlp.yaml
  - sequence_encoder: transformer.yaml
  - embedding: categorical.yaml
  - projection_head: mlp.yaml
  - prediction_head: classification.yaml

_target_: tabular_ssl.models.base.BaseModel

model:
  name: tabular_ssl_model
  type: base
  event_encoder: ${event_encoder}
  sequence_encoder: ${sequence_encoder}
  embedding: ${embedding}
  projection_head: ${projection_head}
  prediction_head: ${prediction_head}
``` 