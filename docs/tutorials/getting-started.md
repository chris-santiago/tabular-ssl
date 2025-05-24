# Getting Started with Tabular SSL

**Time to complete: 10 minutes**

Welcome! This tutorial will get you up and running with Tabular SSL in just a few minutes. You'll explore our interactive demos, understand state-of-the-art corruption strategies, and run your first self-supervised learning experiment.

## What You'll Learn

- How to install and set up Tabular SSL
- How to explore corruption strategies with interactive demos
- How to train with real credit card transaction data
- How to run state-of-the-art SSL experiments (VIME, SCARF, ReConTab)
- Basic concepts of tabular self-supervised learning

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

## Step 2: Explore Corruption Strategies (Interactive Demo)

Before diving into training, let's understand how tabular self-supervised learning works through our interactive demo:

```bash
python demo_corruption_strategies.py
```

This demo shows you:
- **VIME corruption**: Value imputation and mask estimation
- **SCARF corruption**: Contrastive learning with feature swapping
- **ReConTab corruption**: Multi-task reconstruction
- Side-by-side comparison of all approaches

✅ **What to expect**: You'll see how each corruption strategy transforms data, corruption rates, and example outputs.

## Step 3: Try Real Data (Credit Card Demo)

Now let's work with real transaction data:

```bash
python demo_credit_card_data.py
```

This demo:
- Downloads real credit card transaction data from IBM TabFormer
- Shows data preprocessing and sequence creation
- Demonstrates DataModule integration
- Prepares you for actual training

✅ **What to expect**: Download progress, data statistics, and confirmation that everything is ready for training.

## Step 4: Your First SSL Training

Now let's train a state-of-the-art self-supervised model:

```bash
python train.py +experiment=vime_ssl
```

This experiment uses **VIME** (Value Imputation and Mask Estimation):
- Corrupts transaction data by masking features
- Learns to predict which features were masked (mask estimation)
- Learns to reconstruct original values (value imputation)
- Creates powerful representations for downstream tasks

✅ **What to expect**: Training progress with mask estimation and reconstruction losses, checkpoints saved to `outputs/`.

## Step 5: Try Different SSL Methods

Let's experiment with other state-of-the-art approaches:

```bash
# SCARF: Contrastive learning with feature corruption
python train.py +experiment=scarf_ssl

# ReConTab: Multi-task reconstruction
python train.py +experiment=recontab_ssl
```

Each approach uses different corruption strategies:
- **SCARF**: Replaces features with values from other samples (contrastive learning)
- **ReConTab**: Combines masking, noise, and swapping (multi-task reconstruction)

## Step 6: Customize SSL Training

You can easily modify SSL experiments using Hydra's override syntax:

```bash
# Adjust corruption rate for VIME
python train.py +experiment=vime_ssl model/corruption.corruption_rate=0.5

# Change SCARF corruption strategy
python train.py +experiment=scarf_ssl model/corruption.corruption_strategy=marginal_sampling

# Use different sequence length
python train.py +experiment=recontab_ssl data.sequence_length=64

# Adjust learning rate and batch size
python train.py +experiment=vime_ssl model.learning_rate=5e-4 data.batch_size=32
```

## Step 7: Check Your Results

After running experiments, you'll find results in the `outputs/` directory:

```bash
ls outputs/  # See your experiment runs
```

Each run creates a timestamped folder with:
- Configuration files (reproducing exact settings)
- Training logs (TensorBoard, WandB compatible)
- Model checkpoints (best performing models)
- Metrics and plots (loss curves, validation metrics)

For SSL experiments, you'll see specific losses:
- **VIME**: `mask_estimation_loss`, `value_imputation_loss`
- **SCARF**: `contrastive_loss`
- **ReConTab**: `masked_reconstruction`, `denoising`, `unswapping` losses

## Core Concepts Summary

**Corruption Strategies**: Core of self-supervised learning
- **VIME**: Masks features, learns to predict masks and values
- **SCARF**: Swaps features between samples for contrastive learning  
- **ReConTab**: Multi-task corruption (masking + noise + swapping)

**SSL Experiments**: Pre-configured state-of-the-art approaches
- Located in `configs/experiments/`
- Use `+experiment=vime_ssl` (or scarf_ssl, recontab_ssl) to run them

**Model Components**: Modular architecture
- Event encoders (process individual transactions)
- Sequence encoders (model temporal patterns)
- Corruption modules (transform data for SSL)
- Task-specific heads (reconstruction, classification)

**Configuration**: Hydra-based flexible settings
- Override corruption rates: `model/corruption.corruption_rate=0.5`
- Mix components: `model/sequence_encoder=rnn model/corruption=scarf`

## What's Next?

🎯 **Ready for more?** Continue with:
- [Custom Components Tutorial](custom-components.md) - Create your own corruption strategies
- [How-to: SSL Training](../how-to-guides/ssl-training.md) - Advanced self-supervised learning
- [Reference: Corruption Strategies](../reference/corruption-strategies.md) - Technical documentation
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