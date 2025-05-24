# Creating Custom Components

This tutorial will guide you through creating custom components in Tabular SSL using the component registry.

## Prerequisites

- Basic understanding of PyTorch
- Familiarity with the Tabular SSL architecture
- Python environment with Tabular SSL installed

## Introduction

Tabular SSL uses a component registry pattern that allows you to create and register custom components. This enables you to extend the library with your own custom event encoders, sequence encoders, or other components without modifying the core codebase.

## Step 1: Create a Configuration Class

First, define a configuration class for your component. This should inherit from the appropriate base configuration class.

```python
# src/tabular_ssl/models/custom_components.py
from pydantic import Field
from typing import List, Optional

from tabular_ssl.models.base import ComponentConfig

class CustomEncoderConfig(ComponentConfig):
    """Configuration for custom encoder."""
    
    input_dim: int = Field(..., description="Input dimension")
    hidden_dims: List[int] = Field(..., description="Hidden dimensions")
    output_dim: int = Field(..., description="Output dimension")
    dropout: float = Field(0.1, description="Dropout rate")
    use_batch_norm: bool = Field(True, description="Whether to use batch normalization")
    activation: str = Field("relu", description="Activation function")
```

The configuration class uses Pydantic fields with validation and documentation. Required fields are marked with `...` as the default value, while optional fields have explicit defaults.

## Step 2: Create a Component Class

Next, create a component class that inherits from the appropriate base component class:

```python
# src/tabular_ssl/models/custom_components.py
import torch
import torch.nn as nn

from tabular_ssl.models.base import EventEncoder, ComponentRegistry

@ComponentRegistry.register("custom_encoder")
class CustomEncoder(EventEncoder):
    """Custom event encoder implementation."""
    
    def __init__(self, config: CustomEncoderConfig):
        super().__init__(config)
        self.config = config
        
        # Build the layers
        self.layers = nn.ModuleList()
        dims = [config.input_dim] + config.hidden_dims
        
        for i in range(len(dims) - 1):
            layer = []
            layer.append(nn.Linear(dims[i], dims[i + 1]))
            
            if config.use_batch_norm:
                layer.append(nn.BatchNorm1d(dims[i + 1]))
                
            if config.activation == "relu":
                layer.append(nn.ReLU())
            elif config.activation == "leaky_relu":
                layer.append(nn.LeakyReLU(0.2))
            elif config.activation == "gelu":
                layer.append(nn.GELU())
                
            if config.dropout > 0:
                layer.append(nn.Dropout(config.dropout))
                
            self.layers.append(nn.Sequential(*layer))
            
        # Output layer
        self.output_layer = nn.Linear(dims[-1], config.output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size, seq_len, feat_dim = x.shape
        
        # Reshape for processing each timestep
        x = x.reshape(-1, feat_dim)
        
        # Apply the layers
        for layer in self.layers:
            x = layer(x)
            
        # Apply the output layer
        x = self.output_layer(x)
        
        # Reshape back to sequence
        x = x.reshape(batch_size, seq_len, self.config.output_dim)
        
        return x
```

Note the `@ComponentRegistry.register("custom_encoder")` decorator, which registers the component with the registry.

## Step 3: Create a Configuration File

Create a YAML configuration file for your component:

```yaml
# configs/model/event_encoder/custom.yaml
name: custom_encoder
type: custom_encoder
input_dim: 64
hidden_dims: [128, 64]
output_dim: 32
dropout: 0.2
use_batch_norm: true
activation: gelu
```

## Step 4: Use Your Component in an Experiment

Create an experiment configuration that uses your custom component:

```yaml
# configs/experiment/custom_experiment.yaml
# @package _global_

defaults:
  - override /model/event_encoder: custom.yaml
  - override /model/sequence_encoder: transformer.yaml
  - override /trainer: default.yaml
  - override /model: default.yaml
  - _self_

tags: ["custom", "transformer"]

seed: 12345

trainer:
  max_epochs: 50
  gradient_clip_val: 0.5

model:
  optimizer:
    lr: 1.0e-4
    weight_decay: 0.01
```

## Step 5: Import Your Component Module

Make sure your component module is imported somewhere in your code. You can do this in your main training script:

```python
# src/train.py
import hydra
from omegaconf import DictConfig

# Import your custom components
import tabular_ssl.models.custom_components

@hydra.main(config_path="../configs", config_name="config")
def main(config: DictConfig):
    # Rest of your code...
```

## Step 6: Run Your Experiment

Run your experiment with your custom component:

```bash
python src/train.py experiment=custom_experiment
```

## Creating Other Component Types

### Custom Sequence Encoder

Here's how to create a custom sequence encoder:

```python
# Configuration
class CustomSequenceEncoderConfig(ComponentConfig):
    input_dim: int = Field(..., description="Input dimension")
    hidden_dim: int = Field(..., description="Hidden dimension")
    num_layers: int = Field(1, description="Number of layers")
    dropout: float = Field(0.1, description="Dropout rate")
    
# Component
@ComponentRegistry.register("custom_sequence_encoder")
class CustomSequenceEncoder(SequenceEncoder):
    def __init__(self, config: CustomSequenceEncoderConfig):
        super().__init__(config)
        self.config = config
        
        # Implementation...
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass...
        return x
```

### Custom Projection Head

Here's how to create a custom projection head:

```python
# Configuration
class CustomProjectionHeadConfig(ComponentConfig):
    input_dim: int = Field(..., description="Input dimension")
    output_dim: int = Field(..., description="Output dimension")
    
# Component
@ComponentRegistry.register("custom_projection_head")
class CustomProjectionHead(ProjectionHead):
    def __init__(self, config: CustomProjectionHeadConfig):
        super().__init__(config)
        self.config = config
        
        # Implementation...
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass...
        return x
```

## Advanced: Component with Custom Forward Arguments

You can create components with custom forward arguments:

```python
@ComponentRegistry.register("custom_encoder_with_mask")
class CustomEncoderWithMask(EventEncoder):
    def __init__(self, config: CustomEncoderConfig):
        super().__init__(config)
        # Implementation...
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional mask."""
        # Implementation using mask...
        return x
```

## Advanced: Component with Multiple Output Values

Components can return multiple values:

```python
@ComponentRegistry.register("custom_encoder_with_attention")
class CustomEncoderWithAttention(EventEncoder):
    def __init__(self, config: CustomEncoderConfig):
        super().__init__(config)
        # Implementation...
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning encoded values and attention weights."""
        # Implementation...
        return encoded, attention_weights
```

## Advanced: Component with Custom Methods

Components can have custom methods beyond the standard `forward`:

```python
@ComponentRegistry.register("custom_autoencoder")
class CustomAutoEncoder(EventEncoder):
    def __init__(self, config: CustomEncoderConfig):
        super().__init__(config)
        # Implementation...
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (encoding only)."""
        # Implementation...
        return encoded
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        # Implementation...
        return decoded
        
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Full reconstruction."""
        z = self.forward(x)
        return self.decode(z)
```

## Conclusion

You've learned how to create and use custom components in Tabular SSL. The component registry pattern allows you to extend the library with your own implementations while maintaining type safety and configuration validation.

By creating custom components, you can:
- Implement new architectures
- Add domain-specific functionality
- Experiment with novel approaches
- Create specialized components for specific datasets

Remember to always provide proper documentation for your components and to follow the library's design patterns for consistency. 