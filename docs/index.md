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
- [Component Registry](explanation/component-registry.md)
- [SSL Methods](explanation/ssl-methods.md)
- [Performance Considerations](explanation/performance.md)

## Quick Start

```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="config")
def main(config: DictConfig):
    # Create model from configuration
    from tabular_ssl.models.base import BaseModel
    model = BaseModel(config)
    
    # Train model with given config
    trainer = hydra.utils.instantiate(config.trainer)
    trainer.fit(model, datamodule=hydra.utils.instantiate(config.data))
    
    # Make predictions
    trainer.test(model, datamodule=hydra.utils.instantiate(config.data))

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
```

## Key Features

- **Component Registry**: Modular design with type-safe component registration
- **Configuration Management**: Hierarchical configuration with Hydra
- **Self-Supervised Learning**: Multiple SSL methods for tabular data
- **Flexible Architecture**: Mix and match components for custom models
- **Experiment Management**: Easy experiment configuration and tracking
- **Type Safety**: Pydantic-based configuration validation

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 