# Welcome to Tabular SSL Documentation

Welcome to the official documentation for the Tabular SSL library. This documentation is organized following the Diátaxis framework to provide you with the most effective learning and reference experience.

## Documentation Sections

### Tutorials
Learn how to use Tabular SSL through step-by-step guides. Start here if you're new to the library.

- [Getting Started](tutorials/getting-started.md)
- [Basic Usage](tutorials/basic-usage.md)
- [Creating Custom Components](tutorials/custom-components.md)

### How-to Guides
Find practical solutions to specific problems and tasks.

- [Data Preparation](how-to-guides/data-preparation.md)
- [Model Training](how-to-guides/model-training.md)
- [Evaluation](how-to-guides/evaluation.md)
- [Configuring Experiments](how-to-guides/configuring-experiments.md)

### Reference
Detailed technical documentation of the library's components.

- [API Reference](reference/api.md)
- [Models](reference/models.md)
- [Data Utilities](reference/data.md)
- [Utility Functions](reference/utils.md)
- [Configuration](reference/config.md)

### Explanation
Understand the concepts and design decisions behind Tabular SSL.

- [Architecture Overview](explanation/architecture.md)
- [SSL Methods](explanation/ssl-methods.md)
- [Performance Considerations](explanation/performance.md)

## Quick Start

```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(config: DictConfig):
    # Create data module
    datamodule = hydra.utils.instantiate(config.data)
    
    # Create model with Hydra instantiation
    model = hydra.utils.instantiate(config.model)
    
    # Create trainer
    trainer = hydra.utils.instantiate(config.trainer)
    
    # Train model
    trainer.fit(model, datamodule=datamodule)
    
    # Test model
    trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tabular-ssl.git
cd tabular-ssl

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Set PYTHONPATH for imports
export PYTHONPATH=$PWD/src
```

## Key Features

- **Simplified Architecture**: Direct component instantiation with constructor parameters
- **Hydra Configuration**: Clean configuration management with `_target_` specifications
- **Self-Supervised Learning**: Multiple SSL methods for tabular data
- **Flexible Components**: Mix and match components for custom models
- **Experiment Management**: Pre-configured experiments for common architectures
- **Modular Design**: Easy to extend and customize

## Quick Examples

### Running Pre-configured Experiments

```bash
# MLP-only baseline
python train.py +experiment=simple_mlp

# Small transformer model
python train.py +experiment=transformer_small

# Large S4 sequence model
python train.py +experiment=s4_large

# RNN baseline
python train.py +experiment=rnn_baseline
```

### Custom Component Combinations

```bash
# Use specific components
python train.py model/sequence_encoder=transformer model/event_encoder=mlp

# MLP-only (no sequence processing)
python train.py model/sequence_encoder=null

# Custom parameters
python train.py model.learning_rate=1e-4 data.batch_size=128
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 