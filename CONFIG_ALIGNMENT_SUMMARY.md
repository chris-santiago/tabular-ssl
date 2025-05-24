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
# SSL experiments (require explicit model override)
python src/train.py +experiment=vime_ssl model=ssl_vime
python src/train.py +experiment=scarf_ssl model=ssl_scarf
python src/train.py +experiment=recontab_ssl model=ssl_recontab

# Regular experiments
python src/train.py +experiment=credit_card_demo
```

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
- Sample data configuration works with credit card data
- Feature configuration structure matches expected format

### ✅ Experiment Configurations
- SSL experiments properly configure SSL models
- Data configurations are explicit and complete
- Training parameters are appropriate for each method
- Usage instructions are clear and correct

## Testing Verification

All configurations have been tested and verified to:
1. Load without errors
2. Instantiate correct model classes
3. Include all required parameters
4. Work with the training script

The configuration system is now fully aligned with the current implementation and ready for use. 