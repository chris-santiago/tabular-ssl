# Hydra Configuration Alignment Summary

This document summarizes the changes made to align the Hydra configurations with the current implementation.

## Issues Fixed

### 1. Model Configuration Paths
**Problem**: Model configurations were using relative paths in defaults, causing Hydra to fail finding component configs.
**Solution**: Updated all model configs to use absolute paths with `/` prefix.

**Files Changed**:
- `configs/model/default.yaml`: Changed `event_encoder: mlp` to `/event_encoder: mlp`
- All SSL model configs already had correct absolute paths

### 2. Missing Data Configuration
**Problem**: Experiments referenced `sample_data` config that didn't exist.
**Solution**: The `sample_data.yaml` was deleted as it was redundant with existing `credit_card.yaml` config.

### 3. Trainer Configuration Conflicts
**Problem**: Trainer config had duplicate callback definitions (both inline and via separate config).
**Solution**: Removed inline callbacks from trainer config, keeping only the separate callbacks config.

**Files Changed**:
- `configs/trainer/default.yaml`: Removed inline callback definitions

### 4. VIME Corruption Configuration
**Problem**: VIME corruption config had parameters that belonged to the SSLModel, not the corruption class.
**Solution**: Removed `mask_estimation_weight` and `value_imputation_weight` from corruption config.

**Files Changed**:
- `configs/corruption/vime.yaml`: Removed SSL model parameters

### 5. Experiment Configuration Structure
**Problem**: SSL experiments were not properly loading SSL models due to Hydra override precedence issues.
**Solution**: Restructured experiment configs to not use defaults overrides, but instead specify explicit configurations and require explicit model override on command line.

**Files Changed**:
- `configs/experiments/vime_ssl.yaml`
- `configs/experiments/scarf_ssl.yaml` 
- `configs/experiments/recontab_ssl.yaml`
- `configs/experiments/credit_card_demo.yaml`

### 6. Experiment Folder Review and Alignment
**Problem**: Existing experiment configs in `/experiment` folder had broken references to `sample_data` and incomplete configurations.
**Solution**: Completely rewrote all experiment configurations to align with current implementation.

**Files Fixed/Created**:
- `configs/experiment/vime_ssl.yaml`: Fixed VIME SSL experiment with proper parameters
- `configs/experiment/scarf_ssl.yaml`: Created SCARF SSL experiment config  
- `configs/experiment/recontab_ssl.yaml`: Created ReConTab SSL experiment config
- `configs/experiment/quick_vime_ssl.yaml`: Recreated quick testing config
- `configs/experiment/compare_corruptions.yaml`: Fixed multi-run comparison config

## Current Usage Patterns

### Basic Usage
```bash
# Default model (BaseModel)
python src/train.py

# SSL models require explicit model specification
python src/train.py model=ssl_vime
python src/train.py model=ssl_scarf
python src/train.py model=ssl_recontab
```

### Experiment Usage
```bash
# SSL experiments (model is specified in experiment config)
python src/train.py experiment=vime_ssl
python src/train.py experiment=scarf_ssl
python src/train.py experiment=recontab_ssl

# Quick testing experiment
python src/train.py experiment=quick_vime_ssl

# Multi-run comparison of corruption strategies
python src/train.py experiment=compare_corruptions --multirun
```

### Experiment vs Experiments Folders
- `configs/experiment/`: Contains working experiment configurations that use defaults system
- `configs/experiments/`: Contains alternative experiment configurations (some deleted, some fixed)

## Configuration Structure Verification

All configurations now properly align with the implementation:

### ✅ Model Configurations
- `default.yaml`: Uses `BaseModel` with all components
- `ssl_vime.yaml`: Uses `SSLModel` with VIME corruption
- `ssl_scarf.yaml`: Uses `SSLModel` with SCARF corruption  
- `ssl_recontab.yaml`: Uses `SSLModel` with ReConTab corruption

### ✅ Component Configurations
- All event encoders match implementation parameters
- All sequence encoders match implementation parameters
- All corruption strategies match implementation parameters
- All embedding and head configurations are correct

### ✅ Data Configurations
- `TabularDataModule` parameters align with implementation
- Credit card data configuration works properly
- Feature configuration structure matches expected format

### ✅ Experiment Configurations
**Experiment Folder (`configs/experiment/`)**:
- `vime_ssl.yaml`: ✅ VIME SSL with proper `mask_estimation_weight` and `value_imputation_weight`
- `scarf_ssl.yaml`: ✅ SCARF SSL with proper `contrastive_temperature` 
- `recontab_ssl.yaml`: ✅ ReConTab SSL with proper `reconstruction_weights`
- `quick_vime_ssl.yaml`: ✅ Fast testing configuration with reduced epochs/data
- `compare_corruptions.yaml`: ✅ Multi-run setup for comparing corruption strategies

All experiments:
- Use correct SSL model targets (`SSLModel`)
- Include SSL-specific parameters that match source code
- Use proper data configurations with credit card data
- Have appropriate training parameters for each method
- Include proper callback and monitoring configurations

## SSL-Specific Parameter Alignment

### ✅ VIME Parameters
- `mask_estimation_weight`: Controls mask prediction loss weight
- `value_imputation_weight`: Controls value imputation loss weight
- Both parameters verified to exist in `SSLModel.__init__()` and used in loss computation

### ✅ SCARF Parameters  
- `contrastive_temperature`: Temperature parameter for InfoNCE contrastive loss
- Verified to exist in `SSLModel.__init__()` and used in contrastive loss computation

### ✅ ReConTab Parameters
- `reconstruction_weights`: Dictionary with weights for different reconstruction tasks
  - `masked`: Weight for masked token reconstruction
  - `denoising`: Weight for denoising reconstruction
  - `unswapping`: Weight for unswapping reconstruction
- Verified to exist in `SSLModel.__init__()` and used in multi-task loss computation

## Testing Verification

All configurations have been tested and verified to:
1. ✅ Load without errors
2. ✅ Instantiate correct model classes (`SSLModel` for SSL experiments)
3. ✅ Include all required SSL-specific parameters
4. ✅ Work with the training script (`python src/train.py experiment=...`)
5. ✅ Have proper parameter values that align with source code defaults
6. ✅ Use correct data configurations with existing credit card data

The configuration system is now fully aligned with the current implementation. All SSL experiments properly use the `SSLModel` class with their respective corruption strategies and method-specific parameters. The experiment configurations provide ready-to-use setups for each SSL method with appropriate hyperparameters. 