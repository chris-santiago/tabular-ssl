# Configuring Experiments

This guide explains how to configure and run experiments using Hydra in Tabular SSL.

## Introduction

Tabular SSL uses Hydra for configuration management, which enables hierarchical configuration composition, command-line overrides, and experiment tracking. This guide will show you how to:

1. Use the configuration structure
2. Create and run experiments
3. Override default parameters
4. Run parameter sweeps

## Configuration Structure

The configuration files are organized in a hierarchical structure:

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

## Basic Usage

### Running with Default Configuration

To run with the default configuration:

```bash
python src/train.py
```

This will use the configuration in `configs/config.yaml`, which composes configurations from the other directories.

### Overriding Parameters

You can override any parameter using the command line:

```bash
python src/train.py model.optimizer.lr=0.001 trainer.max_epochs=50
```

This will override the learning rate and the maximum number of epochs while using the default values for all other parameters.

### Using a Specific Configuration

You can use a specific configuration for a component:

```bash
python src/train.py model/event_encoder=mlp model/sequence_encoder=transformer
```

This will use the MLP event encoder and Transformer sequence encoder configurations.

## Creating Experiments

### Experiment Configuration Files

Experiment configuration files are stored in `configs/experiment/` and provide a way to group parameter overrides.

Here's an example experiment configuration file:

```yaml
# configs/experiment/transformer_ssl.yaml
# @package _global_

defaults:
  - override /model/event_encoder: mlp.yaml
  - override /model/sequence_encoder: transformer.yaml
  - override /trainer: default.yaml
  - override /model: default.yaml
  - override /callbacks: default.yaml
  - _self_

tags: ["transformer", "ssl"]

seed: 12345

trainer:
  max_epochs: 100
  gradient_clip_val: 0.5

model:
  optimizer:
    lr: 1.0e-4
    weight_decay: 0.01
```

Key things to note:

1. `# @package _global_`: This indicates that the configuration should be merged at the global level
2. `defaults`: Specifies which configurations to use as defaults
3. `override /path/to/config`: Overrides a specific configuration
4. `_self_`: Ensures that the current file's configurations are applied after all others

### Running an Experiment

To run an experiment:

```bash
python src/train.py experiment=transformer_ssl
```

This will use the configuration defined in `configs/experiment/transformer_ssl.yaml`.

### Extending an Experiment

You can extend an experiment by overriding its parameters:

```bash
python src/train.py experiment=transformer_ssl trainer.max_epochs=200
```

This will use the transformer_ssl experiment configuration with the maximum epochs set to 200.

## Debugging

### Debug Mode

You can run in debug mode to speed up debugging:

```bash
python src/train.py debug=true
```

This will typically:
- Run on a smaller dataset
- Use fewer epochs
- Disable certain features like logging

## Experiment Tracking

### Logging and Output

When you run an experiment, Hydra creates an output directory for that run:

```
outputs/
└── 2023-06-15/
    └── 12-34-56/
        ├── .hydra/
        │   ├── config.yaml
        │   ├── hydra.yaml
        │   └── overrides.yaml
        ├── checkpoints/
        └── logs/
```

The `.hydra/` directory contains the full configuration that was used for the run.

### Tags

You can add tags to your experiments:

```yaml
# configs/experiment/transformer_ssl.yaml
tags: ["transformer", "ssl"]
```

Or via the command line:

```bash
python src/train.py tags="[transformer, ssl]"
```

These tags can be used for filtering and grouping experiments.

## Parameter Sweeps

Hydra allows you to perform parameter sweeps by specifying multiple values for a parameter.

### Basic Sweep

```bash
python src/train.py -m model.optimizer.lr=1e-3,1e-4,1e-5
```

This will run three experiments with different learning rates.

### Multi-Parameter Sweep

```bash
python src/train.py -m model.optimizer.lr=1e-3,1e-4 model.optimizer.weight_decay=0.01,0.001
```

This will run 4 experiments (2 learning rates × 2 weight decay values).

### Sweep with Experiment

```bash
python src/train.py -m experiment=transformer_ssl,s4_ssl
```

This will run both the transformer_ssl and s4_ssl experiments.

## Advanced Configuration

### Using Environment Variables

You can use environment variables in your configurations:

```yaml
data:
  path: ${oc.env:DATA_PATH,/default/path}
```

This will use the `DATA_PATH` environment variable if it exists, or fall back to `/default/path`.

### Using Interpolation

You can reference other configuration values:

```yaml
model:
  input_dim: 64
  hidden_dim: ${model.input_dim}  # References input_dim
```

### Dynamic Default Values

You can compute default values based on other parameters:

```yaml
model:
  input_dim: 64
  hidden_dim: ${eval:2 * ${model.input_dim}}  # Dynamic computation
```

## Best Practices

### Naming Conventions

1. Use descriptive names for experiment files
2. Group related parameters together
3. Use consistent naming across configurations

### Configuration Structure

1. Keep configuration files small and focused
2. Use defaults for common parameters
3. Override only what's necessary

### Experiment Management

1. Use meaningful tags for experiments
2. Add a brief description in the experiment file
3. Document key parameter choices

## Typical Workflow

1. Start with an existing experiment: `python src/train.py experiment=transformer_ssl`
2. Make modifications via the command line: `python src/train.py experiment=transformer_ssl model.optimizer.lr=1e-5`
3. If the modifications work well, create a new experiment file
4. Run parameter sweeps to find optimal values: `python src/train.py -m experiment=my_new_experiment model.optimizer.lr=1e-3,1e-4,1e-5`

## Conclusion

Hydra provides a powerful way to configure and track experiments in Tabular SSL. By using experiment configuration files, command-line overrides, and parameter sweeps, you can efficiently explore the parameter space and find the best configurations for your specific task. 