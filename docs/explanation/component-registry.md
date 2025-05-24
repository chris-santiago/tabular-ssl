# Component Registry

This section explains the component registry pattern used in Tabular SSL and its benefits.

## Overview

The component registry is a core architectural pattern in Tabular SSL that enables a modular, extensible, and type-safe approach to building self-supervised learning models for tabular data.

## How the Registry Works

### Registry Implementation

The component registry is implemented as a class with static methods for registering and retrieving components:

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
    
    @classmethod
    def list_components(cls) -> Dict[str, Type['BaseComponent']]:
        """Get a dictionary of all registered components."""
        return cls._components.copy()
```

### Component Registration

Components are registered using a decorator:

```python
@ComponentRegistry.register("mlp_event_encoder")
class MLPEventEncoder(EventEncoder):
    def __init__(self, config: MLPConfig):
        super().__init__(config)
        # Implementation...
```

This decorator adds the component class to the registry with the specified name.

### Component Configuration

Each component has a configuration class that inherits from `ComponentConfig`:

```python
class MLPConfig(ComponentConfig):
    """Configuration for MLP-based components."""
    
    input_dim: int = Field(..., description="Input dimension")
    hidden_dims: List[int] = Field(..., description="Hidden dimensions")
    output_dim: int = Field(..., description="Output dimension")
    dropout: float = Field(0.1, description="Dropout rate")
    use_batch_norm: bool = Field(True, description="Whether to use batch normalization")
```

The configuration class uses Pydantic for validation and includes fields specific to that component.

### Component Instantiation

Components are instantiated from their configuration:

```python
# Get component class
component_cls = ComponentRegistry.get(config.type)

# Instantiate component
component = component_cls(config)
```

## Registry Benefits

### 1. Separation of Concerns

The registry pattern separates several concerns:

- **Components**: Focus on their specific functionality
- **Registry**: Handles component discovery and retrieval
- **Configuration**: Handles component parameters
- **Factory**: Handles component instantiation

This separation makes the codebase easier to understand and modify.

### 2. Type Safety and Validation

The registry pattern enables several layers of validation:

- **Component Type Checking**: Ensuring that components implement the required interfaces
- **Configuration Validation**: Using Pydantic to validate component parameters
- **Registry Existence Checking**: Ensuring that requested components exist

This helps catch configuration errors early.

### 3. Extensibility

The registry pattern makes the codebase highly extensible:

- **Add New Components**: Without modifying existing code
- **Override Existing Components**: By registering a component with the same name
- **User Extensions**: Users can register their own components

This allows for a plugin-like architecture.

### 4. Dynamic Component Loading

The registry pattern enables dynamic component loading:

- **Runtime Component Selection**: Components are loaded based on configuration
- **Lazy Initialization**: Components are only initialized when needed
- **Conditional Components**: Components can be conditionally included or excluded

This flexibility is essential for a library that needs to support many different use cases.

### 5. Improved Testing

The registry pattern improves testability:

- **Mock Components**: Components can be replaced with mocks for testing
- **Component Isolation**: Components can be tested in isolation
- **Configuration Testing**: Configurations can be validated separately

## Registry and Hydra Integration

The component registry integrates with Hydra's configuration system:

### From Hydra Configuration to Component Configuration

```python
def from_hydra(cls, config: DictConfig) -> "ComponentConfig":
    """Create a ComponentConfig from a Hydra config."""
    config_dict = OmegaConf.to_container(config, resolve=True)
    return cls(**config_dict)
```

### Hydra Configuration Example

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

### Component Initialization from Hydra

```python
# In the model class
def __init__(self, config: DictConfig):
    super().__init__()
    self.config = config
    
    # Convert Hydra configs to ComponentConfigs
    self.event_encoder_config = ComponentConfig.from_hydra(config.model.event_encoder)
    
    # Initialize components
    self.event_encoder = self._init_component(self.event_encoder_config)
```

## Component Types

The registry includes several component types:

### EventEncoder

Base class for components that encode individual events or timesteps.

```python
@ComponentRegistry.register("mlp_event_encoder")
class MLPEventEncoder(EventEncoder):
    """MLP-based event encoder."""
```

### SequenceEncoder

Base class for components that encode sequences of events.

```python
@ComponentRegistry.register("transformer")
class TransformerSequenceModel(SequenceEncoder):
    """Transformer sequence encoder."""
```

### EmbeddingLayer

Base class for components that handle embedding of categorical features.

```python
@ComponentRegistry.register("categorical_embedding")
class CategoricalEmbedding(EmbeddingLayer):
    """Embedding layer for categorical variables."""
```

### ProjectionHead

Base class for components that project encoded representations to a different space.

```python
@ComponentRegistry.register("mlp_projection")
class MLPProjectionHead(ProjectionHead):
    """MLP-based projection head."""
```

### PredictionHead

Base class for components that generate predictions from encoded representations.

```python
@ComponentRegistry.register("classification")
class ClassificationHead(PredictionHead):
    """Classification head."""
```

## Best Practices

### When to Register Components

Components should be registered when they:

1. **Provide Reusable Functionality**: Will be used in multiple places
2. **Need Configuration**: Have parameters that can be configured
3. **Represent Alternative Implementations**: Are different ways to implement the same interface

### Naming Components

Component names should:

1. **Be Descriptive**: Clearly indicate what the component does
2. **Use Consistent Naming**: Follow a consistent pattern
3. **Avoid Collisions**: Be unique within the registry

### Documentation

Component documentation should include:

1. **Purpose**: What the component is for
2. **Configuration**: What configuration parameters are available
3. **Usage Examples**: How to use the component
4. **Performance Considerations**: Any performance implications

## Examples

### Implementing a Custom Component

```python
from tabular_ssl.models.base import EventEncoder, ComponentRegistry
from tabular_ssl.models.components import MLPConfig

@ComponentRegistry.register("custom_encoder")
class CustomEncoder(EventEncoder):
    def __init__(self, config: MLPConfig):
        super().__init__(config)
        self.layers = nn.ModuleList()
        
        # Implementation...
        
    def forward(self, x):
        # Forward pass implementation...
        return encoded
```

### Configuration for Custom Component

```yaml
# configs/model/event_encoder/custom.yaml
name: custom_encoder
type: custom_encoder
input_dim: 64
hidden_dims: [128, 64]
output_dim: 32
dropout: 0.1
use_batch_norm: true
```

### Using Custom Component in Experiment

```yaml
# configs/experiment/custom_experiment.yaml
# @package _global_

defaults:
  - override /model/event_encoder: custom.yaml
  - override /model/sequence_encoder: transformer.yaml
  - _self_

# Other experiment settings...
``` 