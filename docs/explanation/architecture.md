# Architecture Overview

This section explains the architecture and design decisions behind Tabular SSL.

## System Design

### High-Level Architecture

The Tabular SSL system consists of several key components:

1. **Data Processing Layer**
   - Data loading and validation
   - Feature preprocessing
   - Data augmentation

2. **Model Layer**
   - Component Registry for modular design
   - Feature embedding
   - Encoder components (Transformer, RNN, LSTM, S4, etc.)
   - Task-specific heads

3. **Training Layer**
   - Self-supervised learning
   - Optimization
   - Monitoring

4. **Configuration Layer**
   - Hydra configuration management
   - Experiment tracking
   - Parameter validation

## Component Registry

One of the core architectural features of Tabular SSL is the Component Registry pattern, which enables a highly modular and extensible design.

### Registry Design

The Component Registry is a central repository that maps component names to their implementations:

```python
class ComponentRegistry:
    """Registry for model components."""
    
    _components: ClassVar[Dict[str, Type['BaseComponent']]] = {}
    
    @classmethod
    def register(cls, name: str) -> Type[T]:
        """Register a component class."""
        def decorator(component_cls: Type[T]) -> Type[T]:
            cls._components[name] = component_cls
            return component_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type['BaseComponent']:
        """Get a component class by name."""
        if name not in cls._components:
            raise KeyError(f"Component {name} not found in registry")
        return cls._components[name]
```

### Component Configuration

Each component has its own configuration class that inherits from `ComponentConfig`:

```python
class ComponentConfig(PydanticBaseModel):
    """Base configuration for components."""
    
    name: str = Field(..., description="Name of the component")
    type: str = Field(..., description="Type of the component")
    
    @validator('type')
    def validate_type(cls, v: str) -> str:
        """Validate that the component type exists in the registry."""
        if v not in ComponentRegistry._components:
            raise ValueError(f"Component type {v} not found in registry")
        return v
```

### Component Initialization

Components are initialized using their configuration:

```python
def _init_component(self, config: ComponentConfig) -> BaseComponent:
    """Initialize a component from its configuration."""
    component_cls = ComponentRegistry.get(config.type)
    return component_cls(config)
```

### Benefits of the Registry Pattern

1. **Modularity**: Components can be added, removed, or replaced independently
2. **Validation**: Configuration is validated before components are initialized
3. **Extensibility**: New components can be added without modifying existing code
4. **Dynamic Loading**: Components are loaded at runtime based on configuration
5. **Type Safety**: Component types are checked during initialization

## Component Details

### Base Components

Tabular SSL defines several base component types:

1. **EventEncoder**: Encodes individual events or timesteps
2. **SequenceEncoder**: Encodes sequences of events
3. **EmbeddingLayer**: Handles embedding of categorical features
4. **ProjectionHead**: Projects encoded representations to a different space
5. **PredictionHead**: Generates predictions from encoded representations

Each component type has multiple implementations that can be selected via configuration.

### Available Components

#### Event Encoders
- `mlp_event_encoder`: MLP-based event encoder
- `autoencoder`: Autoencoder-based event encoder
- `contrastive`: Contrastive learning event encoder

#### Sequence Encoders
- `rnn`: Basic RNN encoder
- `lstm`: LSTM encoder
- `gru`: GRU encoder
- `transformer`: Transformer encoder
- `s4`: Diagonal State Space Model (S4) encoder

#### Embedding Layers
- `categorical_embedding`: Embedding layer for categorical variables

#### Projection Heads
- `mlp_projection`: MLP-based projection head

#### Prediction Heads
- `classification`: Classification head

#### Corruption Strategies
- `random_masking`: Random masking corruption
- `gaussian_noise`: Gaussian noise corruption
- `swapping`: Feature swapping corruption
- `vime`: VIME-style corruption
- `corruption_pipeline`: Pipeline of multiple corruption strategies

## Configuration System

The system uses Hydra's configuration system with structured configuration files:

```
configs/
├── config.yaml                # Main configuration
├── model/                     # Model configurations
│   ├── default.yaml          # Default model config
│   ├── event_encoder/        # Event encoder configs
│   ├── sequence_encoder/     # Sequence encoder configs
│   ├── embedding/            # Embedding configs
│   ├── projection_head/      # Projection head configs
│   └── prediction_head/      # Prediction head configs
├── data/                     # Data configurations
├── trainer/                  # Training configurations
├── callbacks/                # Callback configurations
├── logger/                   # Logger configurations
├── experiment/               # Experiment configurations
├── hydra/                    # Hydra-specific configurations
└── paths/                    # Path configurations
```

### Configuration Composition

Configurations are composed hierarchically:

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

### Experiment Configuration

Experiments override specific parts of the configuration:

```yaml
# configs/experiment/s4_sequence.yaml
# @package _global_

defaults:
  - override /model/sequence_encoder: s4.yaml
  - override /trainer: default.yaml
  - override /model: default.yaml
  - override /callbacks: default.yaml
  - _self_

tags: ["s4", "sequence"]

trainer:
  max_epochs: 50
  gradient_clip_val: 0.5

model:
  optimizer:
    lr: 5.0e-4
    weight_decay: 0.05
```

### Hydra-to-Component Integration

The system translates Hydra configurations to component configurations:

```python
# Convert Hydra configs to ComponentConfigs
self.event_encoder_config = ComponentConfig.from_hydra(config.model.event_encoder)
self.sequence_encoder_config = ComponentConfig.from_hydra(config.model.sequence_encoder)

# Initialize components
self.event_encoder = self._init_component(self.event_encoder_config)
self.sequence_encoder = self._init_component(self.sequence_encoder_config)
```

## Design Decisions

### Why Component Registry?

The Component Registry pattern was chosen for several reasons:

1. **Separation of Concerns**
   - Components focus on their specific functionality
   - Registry handles component discovery and initialization
   - Configuration handles component parameters

2. **Extensibility**
   - New components can be added without modifying existing code
   - Custom components can be registered by users
   - Experiments can mix and match components

3. **Validation**
   - Component types are validated during initialization
   - Configuration parameters are validated using Pydantic
   - Better error messages for misconfiguration

### Why Hydra Configuration?

Hydra provides several benefits for configuration management:

1. **Hierarchical Configuration**
   - Configurations are organized into groups
   - Defaults can be overridden selectively
   - Parameters can be composed from multiple sources

2. **Command-line Overrides**
   - Parameters can be changed at runtime
   - No need to modify configuration files
   - Experiment parameters are explicit

3. **Multirun Capabilities**
   - Parameter sweeps for experimentation
   - Parallel execution of multiple runs
   - Organized output directories

## Implementation Details

### Code Organization

```
src/
├── tabular_ssl/              # Core package
│   ├── data/                # Data loading and processing
│   ├── models/              # Model implementations
│   │   ├── base.py         # Base model and component registry
│   │   ├── components.py   # Model components
│   │   └── s4.py           # S4 implementation
│   └── utils/              # Utility functions
└── train.py                 # Training script
```

### Key Classes

#### ComponentRegistry
- Central registry for all components
- Handles component registration and retrieval
- Ensures type safety

#### BaseComponent
- Abstract base class for all components
- Handles configuration validation
- Defines common interface

#### BaseModel
- Main model class
- Composes components based on configuration
- Handles training and inference

#### ComponentConfig
- Base configuration class
- Uses Pydantic for validation
- Integrates with Hydra configuration

## Performance Considerations

### Component Design

1. **Lazy Initialization**
   - Components are only initialized when needed
   - Configuration is validated early
   - Resources are allocated efficiently

2. **Configuration Caching**
   - Configurations are parsed once
   - Common configurations are reused
   - Reduces memory overhead

3. **Dynamic Component Selection**
   - Only required components are initialized
   - Custom components can be more efficient
   - Allows for hardware-specific optimizations

### Memory Efficiency

1. **Batch Processing**
   - Dynamic batch sizes
   - Gradient accumulation
   - Memory-efficient attention

2. **Model Optimization**
   - Parameter sharing
   - Quantization
   - Pruning

### Training Speed

1. **Hardware Acceleration**
   - GPU support
   - Mixed precision
   - Parallel processing

2. **Optimization**
   - Efficient data loading
   - Cached computations
   - Optimized attention

## Related Resources

- [SSL Methods](ssl-methods.md) - Self-supervised learning approaches
- [Performance Considerations](performance.md) - Optimization and scaling
- [API Reference](../reference/api.md) - Technical documentation 