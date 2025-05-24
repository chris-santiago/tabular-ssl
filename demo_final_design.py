#!/usr/bin/env python3
"""
Demo: Final Consistent Design
Shows the complete, standardized design for easy experimentation.
"""

import torch
from hydra import compose, initialize
from omegaconf import DictConfig
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def demo_component_consistency():
    """Demonstrate consistent component interfaces."""
    print("🔧 Component Interface Consistency")
    print("=" * 60)
    
    from tabular_ssl.models.components import (
        MLPEventEncoder, AutoEncoderEventEncoder, ContrastiveEventEncoder,
        TransformerSequenceEncoder, RNNSequenceEncoder, S4SequenceEncoder,
        MLPProjectionHead, ClassificationHead,
        VIMECorruption, SCARFCorruption, ReConTabCorruption
    )
    
    # All components follow the same patterns
    components = {
        "Event Encoders": [
            ("MLP", MLPEventEncoder(input_dim=64, hidden_dims=[128], output_dim=256)),
            ("Autoencoder", AutoEncoderEventEncoder(input_dim=64, hidden_dims=[128], latent_dim=256)),
            ("Contrastive", ContrastiveEventEncoder(input_dim=64, hidden_dims=[128], output_dim=256))
        ],
        "Sequence Encoders": [
            ("Transformer", TransformerSequenceEncoder(input_dim=256, hidden_dim=256, num_layers=2)),
            ("RNN", RNNSequenceEncoder(input_dim=256, hidden_dim=128, num_layers=2)),
            ("S4", S4SequenceEncoder(input_dim=256, hidden_dim=128, num_layers=2))
        ],
        "Projection Heads": [
            ("MLP", MLPProjectionHead(input_dim=256, hidden_dims=[128], output_dim=64))
        ],
        "Prediction Heads": [
            ("Classification", ClassificationHead(input_dim=64, num_classes=10))
        ],
        "Corruption Strategies": [
            ("VIME", VIMECorruption(corruption_rate=0.3)),
            ("SCARF", SCARFCorruption(corruption_rate=0.6)),
            ("ReConTab", ReConTabCorruption(corruption_rate=0.15))
        ]
    }
    
    # Test data
    x = torch.randn(4, 8, 64)  # (batch, seq, features)
    
    for component_type, component_list in components.items():
        print(f"\n📋 {component_type}:")
        print("-" * 40)
        
        for name, component in component_list:
            print(f"🧩 {name}:")
            
            # All components inherit from BaseComponent
            print(f"   📦 Base class: {component.__class__.__bases__[0].__name__}")
            
            # All have consistent attributes
            if hasattr(component, 'input_dim'):
                print(f"   📏 Input dim: {component.input_dim}")
            if hasattr(component, 'output_dim'):
                print(f"   📐 Output dim: {component.output_dim}")
            
            # Test forward pass (adjust input for different component types)
            try:
                if "Corruption" in component_type:
                    component.train()
                    output = component(x)
                    print(f"   ✅ Forward: {type(output)} with keys {list(output.keys()) if isinstance(output, dict) else 'tensor'}")
                elif "Sequence" in component_type:
                    # Use encoded features for sequence encoders
                    seq_input = torch.randn(4, 8, 256)
                    output = component(seq_input)
                    print(f"   ✅ Forward: {output.shape}")
                elif "Projection" in component_type or "Prediction" in component_type:
                    # Use smaller input for heads
                    head_input = torch.randn(4, component.input_dim if hasattr(component, 'input_dim') else 256)
                    output = component(head_input)
                    print(f"   ✅ Forward: {output.shape}")
                else:
                    output = component(x)
                    print(f"   ✅ Forward: {output.shape}")
            except Exception as e:
                print(f"   ❌ Error: {str(e)[:50]}...")


def demo_configuration_consistency():
    """Demonstrate consistent configuration patterns."""
    print("\n📄 Configuration Consistency")
    print("=" * 60)
    
    # Show the standardized configuration structure
    config_structure = {
        "Component Configs": [
            "configs/event_encoder/mlp.yaml",
            "configs/sequence_encoder/transformer.yaml", 
            "configs/corruption/vime.yaml",
            "configs/projection_head/mlp.yaml",
            "configs/prediction_head/classification.yaml"
        ],
        "Model Configs": [
            "configs/model/ssl_vime.yaml",
            "configs/model/ssl_scarf.yaml",
            "configs/model/base_mlp.yaml",
            "configs/model/transformer_classifier.yaml"
        ],
        "Experiment Configs": [
            "configs/experiment/quick_vime_ssl.yaml",
            "configs/experiment/compare_corruptions.yaml"
        ]
    }
    
    print("📁 Standardized Configuration Structure:")
    print("-" * 40)
    
    for category, configs in config_structure.items():
        print(f"\n🗂️ {category}:")
        for config in configs:
            print(f"   📄 {config}")
    
    print("\n🎯 Key Consistency Features:")
    print("   ✅ All components use _target_ for class specification")
    print("   ✅ All inherit from appropriate base classes")
    print("   ✅ All follow same directory structure")
    print("   ✅ All have standardized parameter names")
    print("   ✅ All support null configurations for optional components")


def demo_easy_experimentation():
    """Demonstrate how easy it is to swap components for experimentation."""
    print("\n🧪 Easy Component Swapping")
    print("=" * 60)
    
    # Show how to easily swap components via configuration
    experiment_variations = [
        {
            "name": "Basic MLP",
            "config": {
                "event_encoder": "mlp",
                "sequence_encoder": "null",
                "corruption": None,
                "model_type": "BaseModel"
            }
        },
        {
            "name": "Transformer + VIME SSL",
            "config": {
                "event_encoder": "mlp", 
                "sequence_encoder": "transformer",
                "corruption": "vime",
                "model_type": "SSLModel"
            }
        },
        {
            "name": "RNN + SCARF SSL",
            "config": {
                "event_encoder": "autoencoder",
                "sequence_encoder": "rnn", 
                "corruption": "scarf",
                "model_type": "SSLModel"
            }
        },
        {
            "name": "S4 + ReConTab SSL",
            "config": {
                "event_encoder": "contrastive",
                "sequence_encoder": "s4",
                "corruption": "recontab", 
                "model_type": "SSLModel"
            }
        }
    ]
    
    print("🔄 Experiment Variations:")
    print("-" * 40)
    
    for variation in experiment_variations:
        print(f"\n🧪 {variation['name']}:")
        config = variation['config']
        
        print(f"   📦 Event Encoder: {config['event_encoder']}")
        print(f"   🔗 Sequence Encoder: {config['sequence_encoder']}")
        print(f"   🎭 Corruption: {config['corruption'] or 'None'}")
        print(f"   🤖 Model Type: {config['model_type']}")
        
        # Show command line for this configuration
        cmd_parts = []
        if config['event_encoder'] != 'mlp':
            cmd_parts.append(f"event_encoder={config['event_encoder']}")
        if config['sequence_encoder'] != 'transformer':
            cmd_parts.append(f"sequence_encoder={config['sequence_encoder']}")
        if config['corruption']:
            cmd_parts.append(f"corruption={config['corruption']}")
        
        cmd = f"python train.py model=ssl_{config['corruption'] or 'base'}"
        if cmd_parts:
            cmd += " " + " ".join(cmd_parts)
        
        print(f"   💻 Command: {cmd}")


def demo_modular_composition():
    """Demonstrate modular composition of models."""
    print("\n🔧 Modular Model Composition")
    print("=" * 60)
    
    from tabular_ssl.models.base import BaseModel, SSLModel
    from tabular_ssl.models.components import (
        MLPEventEncoder, TransformerSequenceEncoder, 
        MLPProjectionHead, ClassificationHead,
        VIMECorruption
    )
    
    # Create components
    event_encoder = MLPEventEncoder(input_dim=64, hidden_dims=[128, 256], output_dim=512)
    sequence_encoder = TransformerSequenceEncoder(input_dim=512, hidden_dim=512, num_layers=2)
    projection_head = MLPProjectionHead(input_dim=512, hidden_dims=[256], output_dim=128)
    prediction_head = ClassificationHead(input_dim=128, num_classes=10)
    corruption = VIMECorruption(corruption_rate=0.3)
    
    print("🧩 Available Components:")
    print(f"   📦 Event Encoder: {event_encoder.__class__.__name__} ({event_encoder.input_dim} → {event_encoder.output_dim})")
    print(f"   🔗 Sequence Encoder: {sequence_encoder.__class__.__name__} ({sequence_encoder.input_dim} → {sequence_encoder.output_dim})")
    print(f"   📐 Projection Head: {projection_head.__class__.__name__}")
    print(f"   🎯 Prediction Head: {prediction_head.__class__.__name__}")
    print(f"   🎭 Corruption: {corruption.__class__.__name__}")
    
    # Different model compositions
    compositions = [
        {
            "name": "Simple Classifier",
            "components": {
                "event_encoder": event_encoder,
                "prediction_head": prediction_head
            },
            "model_class": BaseModel
        },
        {
            "name": "Transformer Classifier", 
            "components": {
                "event_encoder": event_encoder,
                "sequence_encoder": sequence_encoder,
                "projection_head": projection_head,
                "prediction_head": prediction_head
            },
            "model_class": BaseModel
        },
        {
            "name": "SSL Model",
            "components": {
                "event_encoder": event_encoder,
                "sequence_encoder": sequence_encoder,
                "corruption": corruption
            },
            "model_class": SSLModel
        }
    ]
    
    print("\n🔄 Model Compositions:")
    print("-" * 40)
    
    # Test data
    x = torch.randn(2, 4, 64)
    
    for comp in compositions:
        print(f"\n🤖 {comp['name']}:")
        
        # Create model
        model = comp['model_class'](**comp['components'])
        
        # Show architecture
        component_names = list(comp['components'].keys())
        print(f"   🏗️ Architecture: {' → '.join(component_names)}")
        
        # Test forward pass
        try:
            if comp['model_class'] == SSLModel:
                model.train()
                loss = model.training_step(x, 0)
                print(f"   ✅ SSL Training: loss = {loss.item():.4f}")
            else:
                output = model(x)
                print(f"   ✅ Forward: {output.shape}")
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:50]}...")


def demo_configuration_override():
    """Demonstrate configuration override capabilities."""
    print("\n⚙️ Configuration Override Examples")
    print("=" * 60)
    
    override_examples = [
        {
            "description": "Quick experimentation with smaller model",
            "command": "python train.py model=ssl_vime model.event_encoder.hidden_dims=[64,128] model.sequence_encoder.num_layers=2",
            "effect": "Reduces model size for faster iteration"
        },
        {
            "description": "Compare corruption strategies",
            "command": "python train.py -m model=ssl_vime,ssl_scarf,ssl_recontab",
            "effect": "Runs multirun comparison of all SSL strategies"
        },
        {
            "description": "Ablation study: no sequence encoder",
            "command": "python train.py model=ssl_vime sequence_encoder=null",
            "effect": "Tests SSL without sequence modeling"
        },
        {
            "description": "Custom corruption rate",
            "command": "python train.py model=ssl_vime corruption.corruption_rate=0.5",
            "effect": "Increases corruption for harder SSL task"
        },
        {
            "description": "Switch to RNN backbone",
            "command": "python train.py model=ssl_vime sequence_encoder=rnn sequence_encoder.rnn_type=gru",
            "effect": "Uses GRU instead of Transformer"
        }
    ]
    
    print("💻 Command Line Override Examples:")
    print("-" * 40)
    
    for i, example in enumerate(override_examples, 1):
        print(f"\n{i}. {example['description']}:")
        print(f"   Command: {example['command']}")
        print(f"   Effect: {example['effect']}")


if __name__ == "__main__":
    demo_component_consistency()
    demo_configuration_consistency()
    demo_easy_experimentation()
    demo_modular_composition()
    demo_configuration_override()
    
    print("\n" + "=" * 60)
    print("🎉 Final Design Achievements:")
    print("   ✅ Consistent interfaces across all components")
    print("   ✅ Standardized configuration structure")
    print("   ✅ Easy component swapping for experimentation")
    print("   ✅ Modular composition with clear abstractions")
    print("   ✅ Intuitive configuration overrides")
    print("   ✅ Simplified but extensible architecture")
    print("   ✅ Maintains all functionality while improving usability")
    print("\n🚀 Ready for fast, iterative experimentation!") 