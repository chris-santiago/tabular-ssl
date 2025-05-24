# Creating Custom Components

**Time to complete: 20 minutes**

In this tutorial, you'll learn how to create your own custom components for Tabular SSL. We'll start with a simple example and gradually build up your understanding.

## What You'll Learn

- How to create a simple custom event encoder
- How to configure it using YAML files
- How to test and use your custom component
- Basic patterns for extending the library

## Prerequisites

- Completed the [Getting Started](getting-started.md) tutorial
- Basic PyTorch knowledge
- Understanding of neural networks

## The Goal

We'll create a custom event encoder that uses a different activation function and architecture than the default MLP encoder. This will teach you the patterns for creating any type of custom component.

## Step 1: Understanding Component Structure

First, let's look at what makes a component in Tabular SSL:

1. **Inherits from a base class** (like `EventEncoder`)
2. **Has a constructor** with parameters
3. **Implements a `forward` method** for processing data
4. **Can be configured** via YAML files

## Step 2: Create Your First Custom Component

Let's create a simple custom event encoder. Create a new file:

```python
# src/tabular_ssl/models/my_components.py
import torch
import torch.nn as nn
from typing import List

from tabular_ssl.models.base import EventEncoder

class SimpleCustomEncoder(EventEncoder):
    """A simple custom encoder with GELU activation and layer normalization."""
    
    def __init__(
        self, 
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Store parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Create layers
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor."""
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Reshape to process all timesteps at once
        x_flat = x.reshape(-1, self.input_dim)
        
        # Apply our layers
        output_flat = self.layers(x_flat)
        
        # Reshape back to sequence format
        output = output_flat.reshape(batch_size, seq_len, self.output_dim)
        
        return output
```

## Step 3: Test Your Component

Before using it in training, let's test that it works:

```python
# test_my_component.py
import torch
from tabular_ssl.models.my_components import SimpleCustomEncoder

def test_custom_encoder():
    # Create test data
    batch_size, seq_len, input_dim = 8, 10, 32
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Create encoder
    encoder = SimpleCustomEncoder(
        input_dim=32,
        output_dim=64,
        hidden_dim=128
    )
    
    # Test forward pass
    output = encoder(x)
    
    # Check output shape
    expected_shape = (batch_size, seq_len, 64)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    print("✅ Custom encoder test passed!")

if __name__ == "__main__":
    test_custom_encoder()
```

Run this test:
```bash
python test_my_component.py
```

## Step 4: Create a Configuration File

Now create a YAML configuration for your component:

```yaml
# configs/model/event_encoder/simple_custom.yaml
_target_: tabular_ssl.models.my_components.SimpleCustomEncoder

input_dim: 64
output_dim: 256
hidden_dim: 128
dropout: 0.1
```

## Step 5: Use Your Component in an Experiment

Create an experiment that uses your custom component:

```yaml
# configs/experiments/my_custom_experiment.yaml
# @package _global_

defaults:
  - override /model/event_encoder: simple_custom
  - override /model/sequence_encoder: null  # Keep it simple for now
  - override /model/projection_head: mlp
  - override /model/prediction_head: classification

tags: ["custom", "simple"]

model:
  learning_rate: 1.0e-3
  weight_decay: 0.01

trainer:
  max_epochs: 5  # Short run for testing

data:
  batch_size: 32
```

## Step 6: Run Your Custom Experiment

First, make sure Python can find your module:

```python
# Add this to your train.py or create a new script
import tabular_ssl.models.my_components  # This imports your custom components
```

Then run your experiment:

```bash
python train.py +experiment=my_custom_experiment
```

## Step 7: Compare with Default

Let's compare your custom component with the default:

```bash
# Run with default MLP encoder
python train.py +experiment=simple_mlp trainer.max_epochs=5

# Run with your custom encoder  
python train.py +experiment=my_custom_experiment
```

Look at the training logs to see how they perform differently!

## Understanding What You Built

Your custom encoder differs from the default in several ways:

- **GELU activation** instead of ReLU
- **Layer normalization** instead of batch normalization  
- **Simpler architecture** with just one hidden layer
- **Different parameter initialization**

These choices can affect training dynamics and final performance.

## Next Steps: Making It Better

Now that you understand the basics, try these improvements:

### 1. Add Multiple Hidden Layers

```python
class BetterCustomEncoder(EventEncoder):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
        
        self.layers = nn.Sequential(*layers[:-2])  # Remove last activation and dropout
```

### 2. Add Residual Connections

```python
class ResidualCustomEncoder(EventEncoder):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ... process x ...
        residual = x if x.size(-1) == output.size(-1) else None
        if residual is not None:
            output = output + residual
        return output
```

## Key Patterns You've Learned

1. **Inherit from base classes** (`EventEncoder`, `SequenceEncoder`, etc.)
2. **Use `_target_` in YAML** to specify your component class
3. **Test components independently** before integration
4. **Import your modules** so Hydra can find them
5. **Handle tensor shapes carefully** (especially batch and sequence dimensions)

## Common Pitfalls to Avoid

- **Forgetting to import** your custom module
- **Shape mismatches** between components
- **Not testing** before using in experiments
- **Complex components** that are hard to debug

## What's Next?

🎯 **Ready for more advanced patterns?** Check out:
- [How-to: Advanced Component Patterns](../how-to-guides/advanced-components.md)
- [Reference: Component API](../reference/models.md#creating-custom-components)
- [Explanation: Architecture Design](../explanation/architecture.md)

---

**Congratulations!** 🎉 You've created your first custom component. You now understand the core patterns for extending Tabular SSL with your own innovations. 