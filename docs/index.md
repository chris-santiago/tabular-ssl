# Welcome to Tabular SSL Documentation

Welcome to the documentation for **Tabular SSL**, a modular library for self-supervised learning on tabular data with state-of-the-art corruption strategies. This documentation follows the [Diátaxis framework](https://diataxis.fr/) to provide you with the most effective learning and reference experience.

## 🚀 Quick Start

New to Tabular SSL? Try our **interactive demos** to see the library in action:

```bash
# Demo corruption strategies (VIME, SCARF, ReConTab)
python demo_corruption_strategies.py

# Demo with real credit card transaction data
python demo_credit_card_data.py

# Train with state-of-the-art SSL methods
python train.py +experiment=vime_ssl     # VIME approach
python train.py +experiment=scarf_ssl    # SCARF approach  
python train.py +experiment=recontab_ssl # ReConTab approach
```

## 🎭 New: Corruption Strategies

We've implemented corruption strategies from leading tabular SSL papers:

- **🎯 VIME** - Value imputation and mask estimation ([NeurIPS 2020](https://arxiv.org/abs/2006.06775))
- **🌟 SCARF** - Contrastive learning with feature corruption ([arXiv 2021](https://arxiv.org/abs/2106.15147))
- **🔧 ReConTab** - Multi-task reconstruction-based learning

## 🏦 Sample Data

Get started immediately with real transaction data from the [IBM TabFormer](https://github.com/IBM/TabFormer) project - no data preparation needed!

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
- [Corruption Strategies](reference/corruption-strategies.md) - VIME, SCARF, and ReConTab implementations
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

### 🎭 Demo Scripts
```bash
# Interactive corruption strategies demo
python demo_corruption_strategies.py

# Real data demo with credit card transactions
python demo_credit_card_data.py
```

### 🧪 SSL Experiments
```bash
# VIME: Value imputation + mask estimation
python train.py +experiment=vime_ssl

# SCARF: Contrastive learning with feature corruption
python train.py +experiment=scarf_ssl

# ReConTab: Multi-task reconstruction
python train.py +experiment=recontab_ssl
```

### 🔧 Custom Configuration
```bash
# Mix and match components
python train.py model/sequence_encoder=rnn model/event_encoder=mlp

# Use different corruption strategies
python train.py model/corruption=vime model/corruption.corruption_rate=0.5

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

- **🎭 State-of-the-Art Corruption Strategies** - VIME, SCARF, and ReConTab implementations
- **🏦 Ready-to-Use Sample Data** - IBM TabFormer credit card transaction dataset
- **🧩 Modular Architecture** - Mix and match components for custom models
- **⚙️ Hydra Configuration** - Flexible, hierarchical configuration management
- **🧪 Pre-configured SSL Experiments** - VIME, SCARF, and ReConTab ready to run
- **🎬 Interactive Demos** - See corruption strategies in action
- **📊 PyTorch Lightning** - Robust training and evaluation framework

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to get involved.

## Support

- 📝 [GitHub Issues](https://github.com/yourusername/tabular-ssl/issues) - Bug reports and feature requests
- 💬 [Discussions](https://github.com/yourusername/tabular-ssl/discussions) - Questions and community support
- 📧 [Contact](mailto:support@tabular-ssl.org) - Direct support

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 