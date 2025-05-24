**This was generated using Cursor IDE w/agent chat.**  

Notes:
  - Default models (or auto) is pretty terrible.  Code quality is inconsistent; it can feel like you're working with a struggling student. At first things look good, but the code is riddled with errors.
  - Test generation w/default models is even worse.  Most tests are not functional-- they're importing classes/funcs that don't exist in the actual source code. It quickly devolves into and endless loop for error fixes.
  - Using the latest, "thinking" or reasoning models is a much better experience. Claude-4-sonnet cleaned up much of the mess from the "auto" models.

# 🎯 Tabular SSL: Self-Supervised Learning for Tabular Data

A **unified, modular framework** for self-supervised learning on tabular data with **state-of-the-art corruption strategies**, **consistent interfaces**, and **fast experimentation**.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-purple.svg)](https://lightning.ai/)
[![Hydra](https://img.shields.io/badge/Hydra-1.3+-green.svg)](https://hydra.cc/)

## ✨ **Key Features**

🔧 **Consistent Design**: All components follow unified interfaces  
🧩 **Modular Architecture**: Easy to compose and extend  
⚡ **Fast Experimentation**: Swap components with simple config changes  
🎭 **State-of-the-Art SSL**: VIME, SCARF, ReConTab implementations  
🏦 **Ready-to-Use Data**: IBM TabFormer credit card dataset included  
📱 **Interactive Demos**: Understand corruption strategies visually  

---

## 🚀 **Quick Start**

### **Installation**
```bash
git clone https://github.com/yourusername/tabular-ssl.git
cd tabular-ssl
pip install -r requirements.txt
pip install -e .
export PYTHONPATH=$PWD/src
```

### **Interactive Demos**
```bash
# 🎭 Explore corruption strategies 
python demo_corruption_strategies.py

# 🏦 Real credit card transaction data
python demo_credit_card_data.py

# 🔧 Final unified design demo
python demo_final_design.py
```

### **Train Models**
```bash
# 🎯 VIME: Value imputation + mask estimation
python train.py model=ssl_vime

# 🌟 SCARF: Contrastive learning with feature corruption  
python train.py model=ssl_scarf

# 🔧 ReConTab: Multi-task reconstruction learning
python train.py model=ssl_recontab

# 🤖 Simple MLP classifier
python train.py model=base_mlp
```

---

## 🧪 **Easy Experimentation**

### **Component Swapping**
```bash
# Switch to RNN backbone
python train.py model=ssl_vime sequence_encoder=rnn

# Use autoencoder event encoder
python train.py model=ssl_scarf event_encoder=autoencoder

# Remove sequence modeling
python train.py model=ssl_vime sequence_encoder=null

# Custom corruption rate
python train.py model=ssl_vime corruption.corruption_rate=0.5
```

### **Systematic Comparison**
```bash
# Compare all SSL strategies
python train.py -m model=ssl_vime,ssl_scarf,ssl_recontab

# Quick iteration setup
python train.py experiment=quick_vime_ssl

# Full comparison experiment
python train.py experiment=compare_corruptions
```

### **Architecture Variants**
```bash
# Transformer + VIME SSL
python train.py model=ssl_vime sequence_encoder=transformer

# S4 + ReConTab SSL  
python train.py model=ssl_recontab sequence_encoder=s4

# RNN + SCARF SSL
python train.py model=ssl_scarf sequence_encoder=rnn sequence_encoder.rnn_type=gru
```

---

## 🏗️ **Architecture Overview**

### **Unified Component Hierarchy**
```
BaseComponent (Abstract)
├── EventEncoder
│   ├── MLPEventEncoder
│   ├── AutoEncoderEventEncoder
│   └── ContrastiveEventEncoder
├── SequenceEncoder  
│   ├── TransformerSequenceEncoder
│   ├── RNNSequenceEncoder
│   └── S4SequenceEncoder
├── ProjectionHead
│   └── MLPProjectionHead
├── PredictionHead
│   └── ClassificationHead
├── EmbeddingLayer
│   └── CategoricalEmbedding
└── BaseCorruption
    ├── VIMECorruption (NeurIPS 2020)
    ├── SCARFCorruption (arXiv 2021)
    └── ReConTabCorruption
```

### **Model Composition**
```python
# Flexible model composition
SSLModel(
    event_encoder=MLPEventEncoder(...),
    sequence_encoder=TransformerSequenceEncoder(...),
    corruption=VIMECorruption(...)  # ← Type auto-detected
)
```

### **Consistent Interfaces**
All components follow the same patterns:
- **Corruption strategies** return `Dict[str, torch.Tensor]` with `'corrupted'`, `'targets'`, `'mask'`, `'metadata'`
- **All components** expose `input_dim` and `output_dim` properties
- **Auto-detection** eliminates configuration errors

---

## 📁 **Configuration Structure**

### **Standardized Layout**
```
configs/
├── corruption/           # 🎭 VIME, SCARF, ReConTab, etc.
├── event_encoder/        # 📦 MLP, Autoencoder, Contrastive
├── sequence_encoder/     # 🔗 Transformer, RNN, S4, null
├── projection_head/      # 📐 MLP, null
├── prediction_head/      # 🎯 Classification, null  
├── embedding/            # 🔤 Categorical, null
├── model/                # 🤖 Complete model configs
└── experiment/           # 🧪 Experiment templates
```

### **Consistent Pattern**
All component configs follow:
```yaml
# Component Description
_target_: tabular_ssl.models.components.ComponentClass
param1: value1
param2: value2
```

---

## 🎭 **State-of-the-Art Corruption Strategies**

### **VIME** - Value Imputation and Mask Estimation
*From "VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain" (NeurIPS 2020)*

```yaml
# configs/corruption/vime.yaml
_target_: tabular_ssl.models.components.VIMECorruption
corruption_rate: 0.3
categorical_indices: []
numerical_indices: [0, 1, 2, 3]
```

**Features:**
- 🎯 Dual pretext tasks: mask estimation + value imputation
- 🔢 Handles categorical and numerical features differently
- 📊 Returns both corrupted data and mask for training

### **SCARF** - Contrastive Learning with Feature Corruption  
*From "SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption" (arXiv 2021)*

```yaml
# configs/corruption/scarf.yaml
_target_: tabular_ssl.models.components.SCARFCorruption
corruption_rate: 0.6
corruption_strategy: random_swap
```

**Features:**
- 🌟 High corruption rate (60%) for effective contrastive learning
- 🔄 Feature swapping between samples in batch
- 🌡️ Temperature-scaled InfoNCE loss

### **ReConTab** - Multi-task Reconstruction
*Reconstruction-based contrastive learning for tabular data*

```yaml
# configs/corruption/recontab.yaml
_target_: tabular_ssl.models.components.ReConTabCorruption
corruption_rate: 0.15
corruption_types: [masking, noise, swapping]
masking_strategy: random
```

**Features:**
- 🔧 Multiple corruption types: masking, noise injection, swapping
- 📊 Detailed corruption tracking for reconstruction targets
- 🎯 Multi-task learning with specialized heads

---

## 📊 **Sample Data**

### **IBM TabFormer Credit Card Dataset**
```python
from tabular_ssl.data.sample_data import load_credit_card_sample

# Download and load real transaction data
data, info = load_credit_card_sample()
print(f"Loaded {len(data)} transactions")
print(f"Features: {info['feature_names']}")
```

### **Synthetic Transaction Generator**
```python
from tabular_ssl.data.sample_data import generate_sequential_transactions

# Generate synthetic data for experimentation
data = generate_sequential_transactions(
    num_users=1000,
    transactions_per_user=50,
    num_features=10
)
```

---

## 🔧 **Advanced Usage**

### **Custom Component Creation**
```python
from tabular_ssl.models.components import BaseCorruption

class CustomCorruption(BaseCorruption):
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Your corruption logic
        return {
            'corrupted': corrupted_x,
            'targets': x,
            'mask': corruption_mask
        }
```

### **Configuration Override Examples**
```bash
# Hyperparameter sweep
python train.py -m model=ssl_vime \
  corruption.corruption_rate=0.1,0.3,0.5 \
  model.learning_rate=1e-4,5e-4,1e-3

# Architecture ablation
python train.py model=ssl_vime \
  event_encoder.hidden_dims=[64,128] \
  sequence_encoder.num_layers=2

# Custom experiment
python train.py model=ssl_vime \
  event_encoder=autoencoder \
  sequence_encoder=s4 \
  corruption.corruption_rate=0.4
```

### **Modular Composition**
```python
# Mix and match components
model = SSLModel(
    event_encoder=AutoEncoderEventEncoder(...),
    sequence_encoder=S4SequenceEncoder(...),
    projection_head=MLPProjectionHead(...),
    corruption=ReConTabCorruption(...)
)
```

---

## 📚 **Project Structure**

```
tabular-ssl/
├── 📁 configs/                    # Hydra configurations
│   ├── 🎭 corruption/            # Corruption strategies
│   ├── 📦 event_encoder/         # Event encoders
│   ├── 🔗 sequence_encoder/      # Sequence encoders
│   ├── 📐 projection_head/       # Projection heads
│   ├── 🎯 prediction_head/       # Prediction heads
│   ├── 🔤 embedding/             # Embedding layers
│   ├── 🤖 model/                 # Complete models
│   └── 🧪 experiment/            # Experiment configs
├── 📁 src/tabular_ssl/
│   ├── 📊 data/                  # Data loading & sample data
│   ├── 🧠 models/                # Model implementations
│   │   ├── base.py              # Base classes & models
│   │   └── components.py        # All components
│   └── 🛠️ utils/                 # Utilities
├── 🎬 demo_*.py                  # Interactive demos
├── 📖 docs/                      # Documentation
└── ✅ tests/                     # Unit tests
```

---

## 🎯 **Design Principles**

✅ **Consistent Interfaces**: All components follow same patterns  
✅ **Simplified Architecture**: Clean abstractions without complexity  
✅ **Maintained Functionality**: All original features preserved  
✅ **Modular & Extensible**: Easy to add new components  
✅ **Intuitive Configuration**: Logical, well-organized configs  
✅ **Fast Experimentation**: Easy component swapping  

---

## 📈 **Benchmarks & Results**

The framework includes implementations of methods from leading papers:

| Method | Paper | Key Innovation |
|--------|-------|----------------|
| **VIME** | NeurIPS 2020 | Dual pretext tasks for tabular SSL |
| **SCARF** | arXiv 2021 | Contrastive learning with feature corruption |
| **ReConTab** | Custom | Multi-task reconstruction learning |

### **Quick Comparison**
```bash
# Run systematic comparison
python train.py experiment=compare_corruptions

# Results logged to W&B automatically
```

---

## 🚀 **Getting Started Guide**

### **1. Explore Demos**
```bash
python demo_corruption_strategies.py  # Understand corruption methods
python demo_credit_card_data.py       # See real data in action
python demo_final_design.py           # Complete design overview
```

### **2. Train Your First Model**
```bash
python train.py model=ssl_vime         # Start with VIME
```

### **3. Experiment with Components**
```bash
python train.py model=ssl_vime sequence_encoder=rnn
python train.py model=ssl_scarf event_encoder=autoencoder
```

### **4. Create Custom Configurations**
```bash
# Copy and modify existing configs
cp configs/corruption/vime.yaml configs/corruption/my_corruption.yaml
# Edit my_corruption.yaml
python train.py model=ssl_vime corruption=my_corruption
```

---

## 🤝 **Contributing**

We welcome contributions! The modular design makes it easy to:

- **Add new corruption strategies** following `BaseCorruption` interface
- **Implement new encoders** extending base classes
- **Create experiment configurations** in `configs/experiment/`
- **Add new sample datasets** in `src/tabular_ssl/data/`

### **Development Setup**
```bash
git clone https://github.com/yourusername/tabular-ssl.git
cd tabular-ssl
pip install -r requirements.txt
pip install -e .
python -m pytest tests/
```

---

## 📖 **Documentation**

- **🎯 [Design Summary](DESIGN_SUMMARY.md)**: Complete design overview
- **📚 [API Reference](docs/api/)**: Detailed API documentation  
- **🧪 [Experiments Guide](docs/experiments/)**: How to create experiments
- **🔧 [Custom Components](docs/components/)**: Adding new components

---

## 📝 **Citation**

```bibtex
@software{tabular_ssl,
  title={Tabular SSL: A Unified Framework for Self-Supervised Learning on Tabular Data},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/tabular-ssl}
}
```

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **VIME**: [Yoon et al., NeurIPS 2020](https://arxiv.org/abs/2006.06775)
- **SCARF**: [Bahri et al., arXiv 2021](https://arxiv.org/abs/2106.15147)
- **S4**: [Gu et al., ICLR 2022](https://arxiv.org/abs/2111.00396)
- **IBM TabFormer**: [Padhi et al., arXiv 2021](https://arxiv.org/abs/2106.11959)
- **PyTorch Lightning**: [Falcon et al.](https://lightning.ai/)
- **Hydra**: [Facebook Research](https://hydra.cc/)

---

**🎉 Ready for fast, iterative tabular SSL experimentation!** 