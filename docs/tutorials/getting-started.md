# Getting Started with Tabular SSL

**Time to complete: 10 minutes**

Welcome! This tutorial will get you up and running with Tabular SSL in just a few minutes. You'll learn how to install the library, run your first experiment, and understand the basic concepts.

## What You'll Learn

- How to install and set up Tabular SSL
- How to run a pre-configured experiment
- How to modify basic settings
- Basic concepts of the library

## Prerequisites

- Python 3.8+ 
- Basic familiarity with command line

## Step 1: Installation

Let's start by installing Tabular SSL:

```bash
# Clone the repository
git clone https://github.com/yourusername/tabular-ssl.git
cd tabular-ssl

# Install the package
pip install -e .

# Set up the Python path
export PYTHONPATH=$PWD/src
```

✅ **Checkpoint**: Verify your installation works:
```bash
python -c "import tabular_ssl; print('✅ Installation successful!')"
```

## Step 2: Your First Experiment

Now let's run your first experiment using a pre-configured setup:

```bash
python train.py +experiment=simple_mlp
```

This command:
- Runs a simple MLP (Multi-Layer Perceptron) model
- Uses default data and training settings
- Should complete in a few minutes

✅ **What to expect**: You should see training progress logs and the model will save results to an `outputs/` directory.

## Step 3: Understanding What Happened

The experiment you just ran used several key components:

- **Event Encoder**: A neural network that processes individual data points
- **Data Module**: Handles loading and preprocessing your data
- **Trainer**: Manages the training process using PyTorch Lightning

The configuration was loaded from `configs/experiments/simple_mlp.yaml`, which specified all the settings.

## Step 4: Try Different Models

Let's experiment with different model types:

```bash
# Try a transformer-based model
python train.py +experiment=transformer_small

# Try an RNN-based model  
python train.py +experiment=rnn_baseline
```

Each experiment uses a different neural network architecture for processing sequences of tabular data.

## Step 5: Modify Basic Settings

You can easily modify settings using Hydra's override syntax:

```bash
# Change the batch size
python train.py +experiment=simple_mlp data.batch_size=32

# Change the learning rate
python train.py +experiment=simple_mlp model.learning_rate=1e-3

# Use a different data configuration
python train.py +experiment=simple_mlp data=simple
```

## Step 6: Check Your Results

After running experiments, you'll find results in the `outputs/` directory:

```bash
ls outputs/  # See your experiment runs
```

Each run creates a timestamped folder with:
- Configuration files
- Training logs
- Model checkpoints
- Metrics and plots

## Core Concepts Summary

**Experiments**: Pre-configured combinations of models, data, and training settings
- Located in `configs/experiments/`
- Use `+experiment=name` to run them

**Components**: Modular building blocks
- Event encoders (process individual data points)
- Sequence encoders (process sequences)  
- Projection heads (transform representations)

**Configuration**: Uses Hydra for flexible settings
- Override with `key=value` syntax
- Hierarchical structure with defaults

## What's Next?

🎯 **Ready for more?** Continue with:
- [Basic Usage](basic-usage.md) - Learn about data preparation and model customization
- [How-to: Model Training](../how-to-guides/model-training.md) - Detailed training guidance
- [Reference: Models](../reference/models.md) - Complete component documentation

## Troubleshooting

**Import errors?** Make sure PYTHONPATH is set:
```bash
export PYTHONPATH=$PWD/src
```

**CUDA out of memory?** Try smaller batch sizes:
```bash
python train.py +experiment=simple_mlp data.batch_size=16
```

**Need help?** Check our [support resources](../index.md#support) or open an issue on GitHub.

---

**Congratulations!** 🎉 You've successfully run your first Tabular SSL experiments. You're now ready to explore more advanced features and customize the library for your specific needs. 