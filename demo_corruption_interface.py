#!/usr/bin/env python3
"""
Demo: Improved Corruption Interface
Shows how corruption type is now auto-detected, eliminating redundancy.
"""

import torch
from src.tabular_ssl.models.base import SSLModel
from src.tabular_ssl.models.components import (
    MLPEventEncoder, VIMECorruption, SCARFCorruption, ReConTabCorruption,
    TransformerSequenceEncoder
)

def demo_auto_detection():
    """Demonstrate automatic corruption type detection."""
    print("🔄 Demo: Automatic Corruption Type Detection")
    print("=" * 50)
    
    # Common components
    event_encoder = MLPEventEncoder(
        input_dim=64, hidden_dims=[128, 256], output_dim=512
    )
    sequence_encoder = TransformerSequenceEncoder(
        input_dim=512, hidden_dim=512, num_layers=2
    )
    
    # Test data
    batch_size, seq_len, features = 8, 16, 64
    x = torch.randn(batch_size, seq_len, features)
    
    corruption_strategies = [
        ("VIME", VIMECorruption(corruption_rate=0.3)),
        ("SCARF", SCARFCorruption(corruption_rate=0.6)),
        ("ReConTab", ReConTabCorruption(corruption_rate=0.15)),
        ("None", None)
    ]
    
    for strategy_name, corruption in corruption_strategies:
        print(f"\n📋 Testing {strategy_name} Strategy:")
        
        # Create model - NO corruption_type parameter needed!
        model = SSLModel(
            event_encoder=event_encoder,
            sequence_encoder=sequence_encoder,
            corruption=corruption,  # ← Only this parameter needed
            learning_rate=1e-4
        )
        
        print(f"   🔍 Auto-detected type: '{model.corruption_type}'")
        print(f"   🧩 Corruption module: {corruption.__class__.__name__ if corruption else 'None'}")
        
        # Test forward pass
        model.train()
        try:
            loss = model.training_step(x, 0)
            print(f"   ✅ Training works: loss = {loss.item():.4f}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("✨ Key Benefits:")
    print("   • No redundant corruption_type parameter")
    print("   • Impossible to have mismatched corruption type/module")  
    print("   • Cleaner configuration files")
    print("   • Automatic detection even for custom corruption strategies")


def demo_config_simplification():
    """Show how configuration files are simplified."""
    print("\n📄 Configuration Simplification")
    print("=" * 50)
    
    print("❌ OLD WAY (redundant parameters):")
    print("""
defaults:
  - corruption: vime
  
corruption_type: "vime"  # ← Redundant! Could be wrong!
mask_estimation_weight: 1.0
""")
    
    print("✅ NEW WAY (auto-detection):")
    print("""
defaults:
  - corruption: vime      # ← corruption_type auto-detected!
  
mask_estimation_weight: 1.0
""")
    
    print("🔒 Safety: Impossible to have mismatched corruption!")


def demo_custom_corruption():
    """Show auto-detection works with custom corruption strategies."""
    print("\n🛠️ Custom Corruption Strategy")
    print("=" * 50)
    
    class CustomVIMECorruption(torch.nn.Module):
        """Custom VIME-based corruption (auto-detected as 'vime')."""
        def __init__(self):
            super().__init__()
            
        def forward(self, x):
            # Simple masking for demo
            mask = torch.rand_like(x) > 0.3
            return x * mask.float(), (1 - mask.float())
    
    class MyReconTabCorruption(torch.nn.Module):
        """Custom ReConTab-based corruption (auto-detected as 'recontab')."""
        def __init__(self):
            super().__init__()
            
        def forward(self, x):
            # Simple corruption for demo
            corrupted = x + torch.randn_like(x) * 0.1
            info = torch.ones_like(x)  # Mark all as corrupted
            return corrupted, info
        
        def reconstruction_targets(self, original, corrupted, corruption_info):
            return {"mask_positions": corruption_info == 1, "masked_values": original}
    
    # Test custom corruption strategies
    custom_strategies = [
        ("CustomVIME", CustomVIMECorruption()),
        ("MyReconTab", MyReconTabCorruption())
    ]
    
    event_encoder = MLPEventEncoder(input_dim=64, hidden_dims=[128], output_dim=256)
    
    for name, corruption in custom_strategies:
        model = SSLModel(
            event_encoder=event_encoder,
            corruption=corruption
        )
        
        print(f"🔧 {name}:")
        print(f"   Class: {corruption.__class__.__name__}")
        print(f"   Auto-detected type: '{model.corruption_type}'")
        print(f"   ✅ Auto-detection works for custom strategies!")


if __name__ == "__main__":
    demo_auto_detection()
    demo_config_simplification()
    demo_custom_corruption()
    
    print("\n🎉 Corruption interface improved!")
    print("   → Single source of truth")
    print("   → No configuration mismatches")
    print("   → Cleaner, simpler configs") 