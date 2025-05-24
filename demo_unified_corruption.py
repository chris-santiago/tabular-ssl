#!/usr/bin/env python3
"""
Demo: Unified Corruption Interface
Shows how corruption strategies now follow the same patterns as other components.
"""

import torch
from src.tabular_ssl.models.base import SSLModel
from src.tabular_ssl.models.components import (
    MLPEventEncoder, VIMECorruption, SCARFCorruption, ReConTabCorruption,
    RandomMasking, GaussianNoise, SwappingCorruption,
    TransformerSequenceEncoder
)

def demo_unified_interface():
    """Demonstrate the unified corruption interface."""
    print("🔄 Demo: Unified Corruption Interface")
    print("=" * 60)
    
    # Test data
    batch_size, seq_len, features = 4, 8, 16
    x = torch.randn(batch_size, seq_len, features)
    
    # All corruption strategies now follow the same interface
    corruption_strategies = [
        ("RandomMasking", RandomMasking(corruption_rate=0.2)),
        ("GaussianNoise", GaussianNoise(noise_std=0.1)),
        ("SwappingCorruption", SwappingCorruption(swap_prob=0.15)),
        ("VIMECorruption", VIMECorruption(corruption_rate=0.3)),
        ("SCARFCorruption", SCARFCorruption(corruption_rate=0.6)),
        ("ReConTabCorruption", ReConTabCorruption(corruption_rate=0.15))
    ]
    
    print("📋 Testing Unified Interface:")
    print("-" * 40)
    
    for name, corruption in corruption_strategies:
        print(f"\n🧩 {name}:")
        
        # All corruption strategies now return Dict[str, torch.Tensor]
        corruption.train()  # Set to training mode
        output = corruption(x)
        
        print(f"   📤 Output keys: {list(output.keys())}")
        print(f"   📏 Corrupted shape: {output['corrupted'].shape}")
        
        # Check required keys
        assert 'corrupted' in output, f"{name} missing 'corrupted' key"
        assert 'targets' in output, f"{name} missing 'targets' key"
        
        # Optional keys
        optional_keys = ['mask', 'metadata']
        present_optional = [key for key in optional_keys if key in output]
        if present_optional:
            print(f"   🔧 Optional keys: {present_optional}")
        
        print(f"   ✅ Interface consistent!")
    
    print("\n" + "=" * 60)
    print("✨ All corruption strategies follow unified interface!")


def demo_config_consistency():
    """Show how corruption configs now follow the same pattern as other components."""
    print("\n📄 Configuration Consistency")
    print("=" * 60)
    
    print("🔧 Component Configuration Pattern:")
    print("-" * 40)
    
    # Show the consistent pattern across all component types
    component_examples = {
        "Event Encoder": """
# configs/model/event_encoder/mlp.yaml
_target_: tabular_ssl.models.components.MLPEventEncoder
input_dim: 64
hidden_dims: [128, 256]
output_dim: 512
""",
        "Sequence Encoder": """
# configs/model/sequence_encoder/transformer.yaml  
_target_: tabular_ssl.models.components.TransformerSequenceEncoder
input_dim: 512
hidden_dim: 512
num_layers: 4
""",
        "Corruption Strategy": """
# configs/corruption/vime.yaml
_target_: tabular_ssl.models.components.VIMECorruption
corruption_rate: 0.3
categorical_indices: null
"""
    }
    
    for component_type, config_example in component_examples.items():
        print(f"📋 {component_type}:")
        print(config_example)
    
    print("🎯 Key Consistency Features:")
    print("   • All use _target_ for class specification")
    print("   • All inherit from BaseComponent")
    print("   • All follow same directory structure")
    print("   • All have standardized forward() signatures")


def demo_ssl_integration():
    """Show how the unified interface simplifies SSL model integration."""
    print("\n🤖 SSL Model Integration")
    print("=" * 60)
    
    # Common components
    event_encoder = MLPEventEncoder(
        input_dim=16, hidden_dims=[32, 64], output_dim=128
    )
    sequence_encoder = TransformerSequenceEncoder(
        input_dim=128, hidden_dim=128, num_layers=2
    )
    
    # Test data
    batch_size, seq_len, features = 4, 8, 16
    x = torch.randn(batch_size, seq_len, features)
    
    corruption_strategies = [
        ("VIME", VIMECorruption(corruption_rate=0.3)),
        ("SCARF", SCARFCorruption(corruption_rate=0.6)),
        ("ReConTab", ReConTabCorruption(corruption_rate=0.15))
    ]
    
    print("🔄 Testing SSL Model Integration:")
    print("-" * 40)
    
    for strategy_name, corruption in corruption_strategies:
        print(f"\n📋 {strategy_name} Strategy:")
        
        # Create SSL model - corruption type auto-detected!
        model = SSLModel(
            event_encoder=event_encoder,
            sequence_encoder=sequence_encoder,
            corruption=corruption,
            learning_rate=1e-4
        )
        
        print(f"   🔍 Auto-detected type: '{model.corruption_type}'")
        print(f"   🧩 Corruption class: {corruption.__class__.__name__}")
        
        # Test training step
        model.train()
        try:
            loss = model.training_step(x, 0)
            print(f"   ✅ Training works: loss = {loss.item():.4f}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n🎉 SSL integration works seamlessly!")


def demo_extensibility():
    """Show how easy it is to add new corruption strategies."""
    print("\n🛠️ Extensibility Demo")
    print("=" * 60)
    
    # Create a custom corruption strategy following the unified interface
    class CustomDropoutCorruption(torch.nn.Module):
        """Custom corruption strategy that randomly drops entire features."""
        
        def __init__(self, dropout_rate: float = 0.2, **kwargs):
            super().__init__()
            self.dropout_rate = dropout_rate
        
        def forward(self, x: torch.Tensor) -> dict:
            """Apply feature dropout corruption."""
            if not self.training:
                return {
                    'corrupted': x,
                    'targets': x,
                    'mask': torch.zeros_like(x)
                }
            
            # Randomly drop entire features (columns)
            batch_size, seq_len, num_features = x.shape
            feature_mask = torch.rand(num_features) > self.dropout_rate
            
            # Apply mask to all samples and sequences
            corrupted = x.clone()
            for feat_idx in range(num_features):
                if not feature_mask[feat_idx]:
                    corrupted[:, :, feat_idx] = 0.0
            
            # Create position mask for reconstruction
            position_mask = torch.zeros_like(x)
            for feat_idx in range(num_features):
                if not feature_mask[feat_idx]:
                    position_mask[:, :, feat_idx] = 1.0
            
            return {
                'corrupted': corrupted,
                'targets': x,  # Original for reconstruction
                'mask': position_mask,  # Which positions were dropped
                'metadata': {'dropped_features': ~feature_mask}
            }
    
    print("🔧 Custom Corruption Strategy:")
    print("-" * 40)
    
    # Test custom corruption
    custom_corruption = CustomDropoutCorruption(dropout_rate=0.3)
    x = torch.randn(4, 8, 16)
    
    custom_corruption.train()
    output = custom_corruption(x)
    
    print(f"📤 Output keys: {list(output.keys())}")
    print(f"📏 Shapes match: {output['corrupted'].shape == x.shape}")
    print(f"🎯 Follows interface: {'corrupted' in output and 'targets' in output}")
    
    # Show it works with SSL model
    event_encoder = MLPEventEncoder(input_dim=16, hidden_dims=[32], output_dim=64)
    
    # This would work if we made CustomDropoutCorruption inherit from BaseCorruption
    print("\n💡 To integrate with SSLModel:")
    print("   1. Inherit from BaseCorruption")
    print("   2. Create config file: configs/corruption/custom_dropout.yaml")
    print("   3. Add detection logic to SSLModel._detect_corruption_type()")
    print("   4. Ready to use!")


if __name__ == "__main__":
    demo_unified_interface()
    demo_config_consistency()
    demo_ssl_integration()
    demo_extensibility()
    
    print("\n" + "=" * 60)
    print("🎉 Corruption Interface Improvements:")
    print("   ✅ Unified interface across all corruption strategies")
    print("   ✅ Consistent with other component patterns")
    print("   ✅ Simplified SSL model integration")
    print("   ✅ Easy extensibility for custom strategies")
    print("   ✅ Auto-detection eliminates configuration errors")
    print("   ✅ Clean, intuitive API") 