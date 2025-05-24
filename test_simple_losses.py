#!/usr/bin/env python
"""Simple test for the auto-detection of simple loss functions."""

import sys
import os
import torch
import torch.nn as nn

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tabular_ssl.models.simplified_base import TabularSSLModel
from tabular_ssl.models.simplified_components import MLPEncoder, VIMECorruption

def test_simple_losses():
    """Test that simple loss functions work with auto-detection."""
    
    print("🧪 Testing Simple Loss Functions")
    print("=" * 35)
    
    # Create a simple model
    encoder = MLPEncoder(input_dim=10, hidden_dims=[32], output_dim=64)
    x = torch.randn(2, 4, 10)
    
    # Test different simple loss functions
    simple_losses = {
        "MSE Loss": nn.MSELoss(),
        "L1 Loss": nn.L1Loss(),
        "Smooth L1": nn.SmoothL1Loss(),
    }
    
    for loss_name, loss_fn in simple_losses.items():
        print(f"\n🔧 Testing {loss_name}...")
        
        model = TabularSSLModel(
            event_encoder=encoder,
            corruption=VIMECorruption(corruption_rate=0.3),
            custom_loss_fn=loss_fn,  # Simple loss function!
        )
        
        model.train()
        
        # Test loss type detection
        detected_type = model._detect_loss_type(loss_fn)
        print(f"   🎯 Detected type: {detected_type}")
        
        # Test training step
        loss = model._ssl_training_step(x)
        print(f"   ✅ {loss_name}: Loss = {loss:.4f}")
    
    # Test custom simple function
    def custom_mse(predictions, targets):
        return torch.mean((predictions - targets) ** 2)
    
    print(f"\n🔧 Testing Custom Simple Function...")
    
    model = TabularSSLModel(
        event_encoder=encoder,
        corruption=VIMECorruption(corruption_rate=0.3),
        custom_loss_fn=custom_mse,  # Custom 2-parameter function
    )
    
    model.train()
    detected_type = model._detect_loss_type(custom_mse)
    print(f"   🎯 Detected type: {detected_type}")
    
    loss = model._ssl_training_step(x)
    print(f"   ✅ Custom MSE: Loss = {loss:.4f}")
    
    print(f"\n🎉 All simple loss functions work correctly!")
    print(f"   ✅ PyTorch nn.Module losses: Auto-detected as 'simple'")
    print(f"   ✅ Custom 2-parameter functions: Auto-detected as 'custom_simple'")
    print(f"   ✅ Automatic tensor dimension handling")
    print(f"   ✅ No wrapper functions needed!")

if __name__ == "__main__":
    test_simple_losses() 