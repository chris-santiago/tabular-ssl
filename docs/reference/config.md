# Configuration Reference

This reference documents the configuration system and available configuration options in Tabular SSL.

## Configuration System

Tabular SSL uses [Hydra](https://hydra.cc/) for configuration management, with Pydantic for validation.

### Main Configuration File

The main configuration file (`configs/config.yaml`) is the entry point for configuration:

```yaml
# @package _global_

# Default configurations that will be merged
defaults:
  - _self_
  - paths: default.yaml
  - hydra: default.yaml
  - model: default.yaml
  - data: default.yaml
  - trainer: default.yaml
  - callbacks: default.yaml
  - logger: default.yaml
  - experiment: null  # No default experiment
  - debug: null  # No debug mode by default

# Project information
project_name: "tabular-ssl"
project_version: "0.1.0"

# Training parameters
task_name: "ssl"
tags: ["tabular", "ssl"]
seed: 42
debug: false

# Experiment tracking
log_dir: ${paths.log_dir}
checkpoint_dir: ${paths.checkpoint_dir}
```

## Component Configurations

### Model Configuration

The model configuration (`configs/model/default.yaml`) specifies the model architecture:

```yaml
# configs/model/default.yaml
defaults:
  - _self_
  - event_encoder: mlp.yaml
  - sequence_encoder: transformer.yaml
  - embedding: categorical.yaml
  - projection_head: mlp.yaml
  - prediction_head: classification.yaml

_target_: tabular_ssl.models.base.BaseModel

model:
  name: tabular_ssl_model
  type: base
  event_encoder: ${event_encoder}
  sequence_encoder: ${sequence_encoder}
  embedding: ${embedding}
  projection_head: ${projection_head}
  prediction_head: ${prediction_head}

  # Optimizer settings
  optimizer:
    _target_: torch.optim.Adam
    lr: 1.0e-3
    weight_decay: 0.0
    
  # Learning rate scheduler
  lr_scheduler:
    _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
    mode: min
    factor: 0.5
    patience: 5
```

### Event Encoder Configurations

#### MLP Event Encoder

```yaml
# configs/model/event_encoder/mlp.yaml
name: mlp_encoder
type: mlp_event_encoder
input_dim: 64
hidden_dims: [128, 256]
output_dim: 512
dropout: 0.1
use_batch_norm: true
```

#### Autoencoder Event Encoder

```yaml
# configs/model/event_encoder/autoencoder.yaml
name: autoencoder_encoder
type: autoencoder
input_dim: 64
hidden_dims: [128, 64]
output_dim: 32
dropout: 0.1
use_batch_norm: true
use_reconstruction_loss: true
```

#### Contrastive Event Encoder

```yaml
# configs/model/event_encoder/contrastive.yaml
name: contrastive_encoder
type: contrastive
input_dim: 64
hidden_dims: [128, 64]
output_dim: 32
dropout: 0.1
use_batch_norm: true
temperature: 0.07
```

### Sequence Encoder Configurations

#### Transformer Encoder

```yaml
# configs/model/sequence_encoder/transformer.yaml
name: transformer_encoder
type: transformer
input_dim: 512
hidden_dim: 512
num_layers: 4
num_heads: 8
dim_feedforward: 2048
dropout: 0.1
bidirectional: true
```

#### LSTM Encoder

```yaml
# configs/model/sequence_encoder/lstm.yaml
name: lstm_encoder
type: lstm
input_dim: 512
hidden_dim: 512
num_layers: 2
dropout: 0.1
bidirectional: true
```

#### S4 Encoder

```yaml
# configs/model/sequence_encoder/s4.yaml
name: s4_encoder
type: s4
d_model: 512
d_state: 64
dropout: 0.1
bidirectional: true
max_sequence_length: 2048
use_checkpoint: false
```

### Embedding Configurations

```yaml
# configs/model/embedding/categorical.yaml
name: categorical_embedding
type: categorical_embedding
embedding_dims:
  - [5, 8]  # 5 categories, 8-dimensional embedding
  - [3, 4]  # 3 categories, 4-dimensional embedding
dropout: 0.1
```

### Projection Head Configurations

```yaml
# configs/model/projection_head/mlp.yaml
name: mlp_projection
type: mlp_projection
input_dim: 512
hidden_dims: [256]
output_dim: 128
dropout: 0.1
use_batch_norm: true
```

### Prediction Head Configurations

```yaml
# configs/model/prediction_head/classification.yaml
name: classification_head
type: classification
input_dim: 128
num_classes: 2
hidden_dims: [64]
dropout: 0.1
use_batch_norm: true
```

## Data Configurations

```yaml
# configs/data/default.yaml
_target_: tabular_ssl.data.TabularDataModule

data:
  name: default_dataset
  path: ${paths.data_dir}/dataset.csv
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  batch_size: 32
  num_workers: 4
  shuffle: true
  drop_last: false
  normalize: true
  categorical_columns: []
  numerical_columns: []
  target_column: null  # For supervised learning
```

## Trainer Configurations

```yaml
# configs/trainer/default.yaml
_target_: pytorch_lightning.Trainer

trainer:
  accelerator: auto
  strategy: auto
  devices: auto
  num_nodes: 1
  precision: 32
  max_epochs: 100
  min_epochs: 1
  max_steps: -1
  min_steps: null
  limit_train_batches: 1.0
  limit_val_batches: 1.0
  limit_test_batches: 1.0
  limit_predict_batches: 1.0
  fast_dev_run: false
  overfit_batches: 0.0
  val_check_interval: 1.0
  check_val_every_n_epoch: 1
  num_sanity_val_steps: 2
  log_every_n_steps: 50
  enable_checkpointing: true
  enable_progress_bar: true
  enable_model_summary: true
  accumulate_grad_batches: 1
  gradient_clip_val: null
  gradient_clip_algorithm: norm
  deterministic: false
  benchmark: false
  inference_mode: true
  use_distributed_sampler: true
  detect_anomaly: false
  barebones: false
  plugins: null
  sync_batchnorm: false
  reload_dataloaders_every_n_epochs: 0
```

## Callbacks Configurations

```yaml
# configs/callbacks/default.yaml
defaults:
  - _self_

callbacks:
  model_checkpoint:
    _target_: pytorch_lightning.callbacks.ModelCheckpoint
    dirpath: ${checkpoint_dir}
    filename: "epoch_{epoch:03d}-val_loss_{val/loss:.4f}"
    monitor: "val/loss"
    mode: "min"
    save_last: true
    save_top_k: 3
    auto_insert_metric_name: false
    
  early_stopping:
    _target_: pytorch_lightning.callbacks.EarlyStopping
    monitor: "val/loss"
    patience: 10
    mode: "min"
    min_delta: 0.0001
    
  lr_monitor:
    _target_: pytorch_lightning.callbacks.LearningRateMonitor
    logging_interval: "epoch"
```

## Logger Configurations

```yaml
# configs/logger/default.yaml
defaults:
  - _self_

logger:
  tensorboard:
    _target_: pytorch_lightning.loggers.TensorBoardLogger
    save_dir: ${log_dir}
    name: null
    version: null
    log_graph: false
    default_hp_metric: true
    prefix: ""
```

## Experiment Configurations

Experiment configurations are stored in `configs/experiment/` and can override any of the above configurations.

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

## Debug Configurations

Debug configurations provide settings for development and debugging:

```yaml
# configs/debug/default.yaml
# @package _global_

# Enable debug mode
debug: true

# Reduce dataset size for faster iterations
data:
  train_ratio: 0.05
  val_ratio: 0.05
  test_ratio: 0.05
  batch_size: 8
  num_workers: 0

# Reduce training time
trainer:
  max_epochs: 5
  limit_train_batches: 10
  limit_val_batches: 10
  limit_test_batches: 10
  log_every_n_steps: 1
  num_sanity_val_steps: 0

# Disable checkpointing
callbacks:
  model_checkpoint:
    save_top_k: 1
    every_n_epochs: 1
```

## Environment Variables

The configuration system supports environment variables using the `${oc.env:VAR_NAME,default_value}` syntax.

For example:

```yaml
paths:
  data_dir: ${oc.env:DATA_DIR,${project_path}/data}
```

## Command-line Overrides

Any configuration parameter can be overridden from the command line:

```bash
python src/train.py model.optimizer.lr=0.001 trainer.max_epochs=50
```

## Multi-run (Parameter Sweeps)

Multiple runs with different parameters can be executed using the `-m` flag:

```bash
python src/train.py -m model.optimizer.lr=1e-3,1e-4,1e-5
```

## See Also

- [Configuring Experiments](../how-to-guides/configuring-experiments.md): Guide for creating and running experiments
- [Hydra Documentation](https://hydra.cc/docs/intro): Official Hydra documentation 