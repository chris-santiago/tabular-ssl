#!/usr/bin/env python
"""Demo showing the unified interface for arbitrary loss functions.

This demonstrates how you can pass ANY loss function to TabularSSLModel:
- Simple losses like nn.MSELoss() 
- Complex SSL losses with full signature
- Custom losses that use whatever parameters they need
"""

import sys
import os
import torch
import torch.nn as nn

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tabular_ssl.models.simplified_base import TabularSSLModel
from tabular_ssl.models.simplified_components import (
    MLPEncoder, VIMECorruption, SCARFCorruption, ReConTabCorruption
)

def demo_unified_loss_interface():
    """Demonstrate the unified loss function interface."""
    
    print("🎯 Unified Loss Function Interface Demo")
    print("=" * 45)
    
    # Create sample components
    encoder = MLPEncoder(input_dim=10, hidden_dims=[32], output_dim=64)
    x = torch.randn(4, 6, 10)
    
    print("\n📋 All loss functions use the same interface:\n")
    
    # =============================================================================
    # 1. SIMPLE PYTORCH LOSSES (fallback to 2-parameter call)
    # =============================================================================
    
    print("1️⃣ Simple PyTorch Losses")
    print("-" * 25)
    
    # These will automatically fallback to simple (predictions, targets) call
    simple_losses = [
        ("MSE Loss", nn.MSELoss()),
        ("L1 Loss", nn.L1Loss()),
        ("Smooth L1", nn.SmoothL1Loss()),
    ]
    
    for loss_name, loss_fn in simple_losses:
        print(f"\n🔧 Testing {loss_name}...")
        
        model = TabularSSLModel(
            event_encoder=encoder,
            corruption=VIMECorruption(corruption_rate=0.3),
            custom_loss_fn=loss_fn  # Just pass it directly!
        )
        
        model.train()
        loss = model._ssl_training_step(x)
        print(f"   ✅ {loss_name}: Loss = {loss:.4f}")
        print(f"   📝 Usage: custom_loss_fn=nn.{loss_fn.__class__.__name__}()")
    
    # =============================================================================
    # 2. COMPLEX SSL LOSSES (use full signature)
    # =============================================================================
    
    print(f"\n2️⃣ Complex SSL Losses")
    print("-" * 25)
    
    def vime_custom_loss(predictions, targets, model, corrupted_data, ssl_loss_weights, **kwargs):
        """Custom VIME loss that uses the full signature."""
        # Ensure model has required heads
        if not hasattr(model, 'mask_head'):
            repr_dim = predictions.size(-1)
            model.mask_head = nn.Linear(repr_dim, 1).to(predictions.device)
        if not hasattr(model, 'value_head'):
            repr_dim = predictions.size(-1)
            model.value_head = nn.Linear(repr_dim, targets.size(-1)).to(predictions.device)
        
        # Predictions
        mask_pred = torch.sigmoid(model.mask_head(predictions))
        value_pred = model.value_head(predictions)
        
        # Get weights
        mask_weight = ssl_loss_weights.get('mask_estimation', 1.0)
        value_weight = ssl_loss_weights.get('value_imputation', 1.0)
        
        # Losses
        mask_true = corrupted_data['mask'].float()
        if mask_true.dim() == 3:
            mask_true = mask_true.mean(dim=-1)
        if mask_pred.dim() == 3:
            mask_pred = mask_pred.squeeze(-1)
        
        mask_loss = nn.functional.binary_cross_entropy(mask_pred, mask_true)
        value_loss = nn.functional.mse_loss(value_pred, targets)
        
        return mask_weight * mask_loss + value_weight * value_loss
    
    def simple_reconstruction_loss(predictions, targets, **kwargs):
        """Simple reconstruction loss that ignores extra parameters."""
        # Create a simple reconstruction head
        if not hasattr(simple_reconstruction_loss, 'head'):
            simple_reconstruction_loss.head = nn.Linear(predictions.size(-1), targets.size(-1)).to(predictions.device)
        
        reconstructed = simple_reconstruction_loss.head(predictions)
        return nn.functional.mse_loss(reconstructed, targets)
    
    def contrastive_loss(predictions, targets, model, corrupted_data, **kwargs):
        """Contrastive loss that uses some of the available parameters."""
        # Normalize representations
        norm_pred = nn.functional.normalize(predictions, dim=-1)
        
        if norm_pred.dim() == 3:  # Handle sequence data
            norm_pred = norm_pred.mean(dim=1)  # Average pooling
        
        # Simple contrastive loss
        batch_size = norm_pred.size(0)
        sim_matrix = torch.matmul(norm_pred, norm_pred.t()) / 0.1  # temperature
        
        # Remove self-similarities
        mask = torch.eye(batch_size, device=norm_pred.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
        
        # InfoNCE-like loss
        labels = torch.arange(batch_size, device=norm_pred.device)
        return nn.functional.cross_entropy(sim_matrix, labels)
    
    complex_losses = [
        ("Custom VIME", vime_custom_loss, VIMECorruption(corruption_rate=0.3)),
        ("Simple Reconstruction", simple_reconstruction_loss, ReConTabCorruption(corruption_rate=0.15)),
        ("Contrastive", contrastive_loss, SCARFCorruption(corruption_rate=0.6)),
    ]
    
    for loss_name, loss_fn, corruption in complex_losses:
        print(f"\n🔧 Testing {loss_name}...")
        
        model = TabularSSLModel(
            event_encoder=encoder,
            corruption=corruption,
            custom_loss_fn=loss_fn,  # Uses full signature
            ssl_loss_weights={'mask_estimation': 1.0, 'value_imputation': 1.0}
        )
        
        model.train()
        loss = model._ssl_training_step(x)
        print(f"   ✅ {loss_name}: Loss = {loss:.4f}")
        print(f"   📝 Usage: custom_loss_fn={loss_fn.__name__}")
    
    # =============================================================================
    # 3. BUILT-IN SSL METHODS (no custom loss needed)
    # =============================================================================
    
    print(f"\n3️⃣ Built-in SSL Methods")
    print("-" * 25)
    
    builtin_methods = [
        ("VIME", VIMECorruption(corruption_rate=0.3)),
        ("SCARF", SCARFCorruption(corruption_rate=0.6)),
        ("ReConTab", ReConTabCorruption(corruption_rate=0.15)),
    ]
    
    for method_name, corruption in builtin_methods:
        print(f"\n🔧 Testing {method_name}...")
        
        model = TabularSSLModel(
            event_encoder=encoder,
            corruption=corruption,
            # No custom_loss_fn - uses built-in method
        )
        
        model.train()
        loss = model._ssl_training_step(x)
        print(f"   ✅ {method_name}: Loss = {loss:.4f}")
        print(f"   📝 Usage: corruption={corruption.__class__.__name__}() (built-in)")
    
    # =============================================================================
    # 4. INTERFACE SUMMARY
    # =============================================================================
    
    print(f"\n📊 Unified Interface Summary")
    print("=" * 45)
    
    summary = """
    🎯 Standard Interface:
    
    custom_loss_fn(
        predictions=representations,    # Model representations
        targets=original_data,         # Original uncorrupted data  
        model=model,                   # Full model access
        corrupted_data=corruption_info, # Corruption details
        ssl_loss_weights=weights,      # Loss component weights
        **kwargs                       # Future extensibility
    )
    
    ✨ Benefits:
    ✅ Single interface for ANY loss function
    ✅ Simple losses automatically fallback to (predictions, targets)
    ✅ Complex losses can use whatever parameters they need
    ✅ Built-in SSL methods work without custom_loss_fn
    ✅ Full backward compatibility
    ✅ Easy to test and debug
    
    📝 Usage Examples:
    • Simple:     custom_loss_fn=nn.MSELoss()
    • Complex:    custom_loss_fn=my_ssl_loss_function  
    • Built-in:   corruption=VIMECorruption() (no custom_loss_fn)
    """
    
    print(summary)
    
    print(f"\n🏆 Perfect! The unified interface is:")
    print(f"   • Simple: Just pass any loss function")
    print(f"   • Flexible: Works with any signature")  
    print(f"   • Powerful: Full access to all information")
    print(f"   • Clean: No complex auto-detection needed")

if __name__ == "__main__":
    demo_unified_loss_interface() 