# Welcome to Tabular SSL Documentation

Welcome to the documentation for **Tabular SSL**, a modular library for self-supervised learning on tabular data. This documentation follows the [Diátaxis framework](https://diataxis.fr/) to provide you with the most effective learning and reference experience.

## Getting Started

New to Tabular SSL? Start with our [Getting Started tutorial](tutorials/getting-started.md) to learn the basics in just a few minutes.

```bash
# Quick start
python train.py +experiment=simple_mlp
```

## Documentation Structure

### 📚 [Tutorials](tutorials/)
**Learning-oriented guides for newcomers**

Step-by-step lessons to help you learn Tabular SSL fundamentals. Start here if you're new to the library.

- [Getting Started](tutorials/getting-started.md) - Your first steps with Tabular SSL
- [Basic Usage](tutorials/basic-usage.md) - Core concepts and workflows
- [Custom Components](tutorials/custom-components.md) - Creating your own components

### 🛠️ [How-to Guides](how-to-guides/)
**Problem-oriented solutions for specific tasks**

Practical guides for accomplishing specific goals and solving real problems.

- [Data Preparation](how-to-guides/data-preparation.md) - Prepare your datasets
- [Model Training](how-to-guides/model-training.md) - Train models effectively
- [Evaluation](how-to-guides/evaluation.md) - Evaluate model performance
- [Configuring Experiments](how-to-guides/configuring-experiments.md) - Set up experiments

### 📖 [Reference](reference/)
**Information-oriented technical documentation**

Complete and accurate technical reference for the library's components and APIs.

- [API Reference](reference/api.md) - Complete API documentation
- [Models](reference/models.md) - Available model components
- [Data](reference/data.md) - Data handling utilities
- [Configuration](reference/config.md) - Configuration system
- [Utilities](reference/utils.md) - Helper functions

### 💡 [Explanation](explanation/)
**Understanding-oriented discussions of key topics**

Background information and conceptual explanations to help you understand the library's design and principles.

- [Architecture Overview](explanation/architecture.md) - System design and principles
- [SSL Methods](explanation/ssl-methods.md) - Self-supervised learning approaches
- [Performance](explanation/performance.md) - Optimization and best practices

## Quick Examples

### Run Pre-configured Experiments
```bash
# Simple MLP baseline
python train.py +experiment=simple_mlp

# Transformer for sequences
python train.py +experiment=transformer_small

# Efficient S4 model
python train.py +experiment=s4_large
```

### Custom Configuration
```bash
# Mix and match components
python train.py model/sequence_encoder=rnn model/event_encoder=mlp

# Adjust hyperparameters
python train.py model.learning_rate=1e-3 data.batch_size=64
```

## Installation

```bash
git clone https://github.com/yourusername/tabular-ssl.git
cd tabular-ssl
pip install -e .
export PYTHONPATH=$PWD/src
```

## Key Features

- **🧩 Modular Architecture** - Mix and match components for custom models
- **⚙️ Hydra Configuration** - Flexible, hierarchical configuration management
- **🔄 Self-Supervised Learning** - Multiple SSL methods for tabular data
- **🚀 Pre-configured Experiments** - Ready-to-use model configurations
- **📊 PyTorch Lightning** - Robust training and evaluation framework

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to get involved.

## Support

- 📝 [GitHub Issues](https://github.com/yourusername/tabular-ssl/issues) - Bug reports and feature requests
- 💬 [Discussions](https://github.com/yourusername/tabular-ssl/discussions) - Questions and community support
- 📧 [Contact](mailto:support@tabular-ssl.org) - Direct support

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 