# Creating Custom Components

This tutorial will guide you through creating custom components in Tabular SSL using the simplified modular architecture.

## Prerequisites

- Basic understanding of PyTorch
- Familiarity with the Tabular SSL architecture
- Python environment with Tabular SSL installed

## Introduction

Tabular SSL uses a simplified modular architecture that allows you to create custom components by directly subclassing base classes and using Hydra's `_target_` mechanism for instantiation. This approach is much simpler than the previous registry-based system.

## Step 1: Create a Custom Component Class

Create a component class that inherits from the appropriate base class:

```python
# src/tabular_ssl/models/custom_components.py
import torch
import torch.nn as nn
from typing import List

from tabular_ssl.models.base import EventEncoder

class CustomEventEncoder(EventEncoder):
    """Custom event encoder implementation."""
    
    def __init__(
        self, 
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        
        # Build the layers
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layer = []
            layer.append(nn.Linear(dims[i], dims[i + 1]))
            
            if use_batch_norm:
                layer.append(nn.BatchNorm1d(dims[i + 1]))
                
            if activation == "relu":
                layer.append(nn.ReLU())
            elif activation == "leaky_relu":
                layer.append(nn.LeakyReLU(0.2))
            elif activation == "gelu":
                layer.append(nn.GELU())
                
            if dropout > 0:
                layer.append(nn.Dropout(dropout))
                
            self.layers.append(nn.Sequential(*layer))
            
        # Output layer
        self.output_layer = nn.Linear(dims[-1], output_dim)
        
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
        x = x.reshape(batch_size, seq_len, self.output_dim)
        
        return x
```

## Step 2: Create a Configuration File

Create a YAML configuration file for your component:

```yaml
# configs/model/event_encoder/custom.yaml
_target_: tabular_ssl.models.custom_components.CustomEventEncoder

input_dim: 64
hidden_dims: [128, 64]
output_dim: 32
dropout: 0.2
activation: gelu
use_batch_norm: true
```

## Step 3: Use Your Component in an Experiment

Create an experiment configuration that uses your custom component:

```yaml
# configs/experiments/custom_experiment.yaml
# @package _global_

defaults:
  - override /model/event_encoder: custom
  - override /model/sequence_encoder: transformer
  - override /model/projection_head: mlp
  - override /model/prediction_head: classification

tags: ["custom", "transformer"]

model:
  learning_rate: 1.0e-4
  weight_decay: 0.01
  optimizer_type: adamw

trainer:
  max_epochs: 50
  gradient_clip_val: 0.5

data:
  batch_size: 64
  sequence_length: 32
```

## Step 4: Import Your Component Module

Make sure your component module is imported. You can do this in your main training script or by adding it to the package's `__init__.py`:

```python
# src/train.py
import hydra
from omegaconf import DictConfig

# Import your custom components module to register them
import tabular_ssl.models.custom_components

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(config: DictConfig):
    # Create components using Hydra
    datamodule = hydra.utils.instantiate(config.data)
    model = hydra.utils.instantiate(config.model)
    trainer = hydra.utils.instantiate(config.trainer)
    
    # Training loop
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
```

## Step 5: Run Your Experiment

Run your experiment with your custom component:

```bash
python train.py +experiment=custom_experiment
```

## Creating Other Component Types

### Custom Sequence Encoder

Here's how to create a custom sequence encoder:

```python
# src/tabular_ssl/models/custom_components.py
from tabular_ssl.models.base import SequenceEncoder

class CustomSequenceEncoder(SequenceEncoder):
    """Custom sequence encoder implementation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Example: Multi-layer GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # x shape: (batch_size, seq_len, input_dim)
        output, _ = self.gru(x)
        # output shape: (batch_size, seq_len, hidden_dim)
        return output
```

Configuration file:

```yaml
# configs/model/sequence_encoder/custom_gru.yaml
_target_: tabular_ssl.models.custom_components.CustomSequenceEncoder

input_dim: 512
hidden_dim: 256
num_layers: 2
dropout: 0.1
```

### Custom Projection Head

```python
# src/tabular_ssl/models/custom_components.py
from tabular_ssl.models.base import ProjectionHead

class CustomProjectionHead(ProjectionHead):
    """Custom projection head with residual connections."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        x = self.input_proj(x)
        
        for block in self.blocks:
            residual = x
            x = block(x) + residual
            
        x = self.output_proj(x)
        return x
```

## Advanced Example: Custom Attention Mechanism

Here's a more complex example with a custom attention mechanism:

```python
# src/tabular_ssl/models/custom_components.py
class CustomAttentionEncoder(SequenceEncoder):
    """Custom encoder with learnable attention mechanism."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention and residual connections."""
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x
```

## Testing Your Custom Component

Create a simple test to verify your component works:

```python
# test_custom_component.py
import torch
from tabular_ssl.models.custom_components import CustomEventEncoder

def test_custom_encoder():
    batch_size, seq_len, input_dim = 32, 16, 64
    
    encoder = CustomEventEncoder(
        input_dim=input_dim,
        hidden_dims=[128, 256],
        output_dim=512,
        dropout=0.1
    )
    
    # Create test input
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    output = encoder(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, 512)
    print("✅ Custom encoder test passed!")

if __name__ == "__main__":
    test_custom_encoder()
```

## Best Practices

1. **Inherit from Base Classes**: Always inherit from appropriate base classes (`EventEncoder`, `SequenceEncoder`, etc.)

2. **Use Type Hints**: Add proper type hints for better code clarity and IDE support

3. **Document Parameters**: Use clear docstrings to document your component's parameters and behavior

4. **Test Thoroughly**: Create unit tests for your custom components

5. **Follow Naming Conventions**: Use descriptive names for your components and configuration files

6. **Handle Edge Cases**: Consider sequence length variations and batch size differences

## Common Issues and Solutions

### Import Errors

Make sure your custom module is importable:

```bash
# Check PYTHONPATH
export PYTHONPATH=$PWD/src

# Test import
python -c "from tabular_ssl.models.custom_components import CustomEventEncoder; print('✅ Import successful')"
```

### Configuration Errors

Validate your configuration:

```bash
# Test configuration loading
python -c "import hydra; hydra.initialize(config_path='configs', version_base=None); cfg = hydra.compose(config_name='config', overrides=['+experiment=custom_experiment']); print('✅ Config valid')"
```

### Dimension Mismatches

Ensure input/output dimensions match between components:

```python
# Check component compatibility
event_encoder_output = 512
sequence_encoder_input = 512  # Should match

assert event_encoder_output == sequence_encoder_input
```

## Next Steps

- Explore the [API Reference](../reference/api.md) for more base classes
- Check out [Architecture Overview](../explanation/architecture.md) for design patterns
- See [Configuration Reference](../reference/config.md) for advanced configuration options 