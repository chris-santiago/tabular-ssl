#!/usr/bin/env python
"""Demo showing how to use simple loss functions with TabularSSLModel.

This demonstrates the simplified interface where users can just pass
nn.MSELoss(), nn.L1Loss(), or any other simple loss function without
needing to write complex wrapper functions.
"""

import sys
import os
import torch
import torch.nn as nn

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tabular_ssl.models.simplified_base import TabularSSLModel, get_ssl_loss_function
from tabular_ssl.models.simplified_components import (
    MLPEncoder, TransformerEncoder, MLPHead,
    VIMECorruption, SCARFCorruption, ReConTabCorruption
)

def demo_simple_loss_functions():
    """Demonstrate using simple loss functions with SSL models."""
    
    print("🎯 Simple Loss Functions Demo")
    print("=" * 40)
    
    # Create sample components
    event_encoder = MLPEncoder(input_dim=10, hidden_dims=[32], output_dim=64)
    sequence_encoder = TransformerEncoder(input_dim=64, hidden_dim=32, num_heads=2, num_layers=1)
    
    # Sample data
    x = torch.randn(4, 6, 10)
    
    print("\n📋 Testing Different Loss Function Types:\n")
    
    # =============================================================================
    # 1. SIMPLE PYTORCH LOSSES
    # =============================================================================
    
    print("1️⃣ Simple PyTorch Losses")
    print("-" * 25)
    
    simple_losses = {
        "MSE Loss": nn.MSELoss(),
        "L1 Loss": nn.L1Loss(), 
        "Smooth L1": nn.SmoothL1Loss(),
        "Huber Loss": nn.HuberLoss(delta=1.0),
    }
    
    for loss_name, loss_fn in simple_losses.items():
        print(f"\n🔧 Testing {loss_name}...")
        
        model = TabularSSLModel(
            event_encoder=event_encoder,
            sequence_encoder=sequence_encoder,
            corruption=VIMECorruption(corruption_rate=0.3),
            custom_loss_fn=loss_fn,  # Just pass the loss directly!
            learning_rate=1e-3
        )
        
        model.train()
        loss = model._ssl_training_step(x)
        print(f"   ✅ {loss_name}: Loss = {loss:.4f}")
        print(f"   📝 Usage: custom_loss_fn=nn.{loss_fn.__class__.__name__}()")
    
    # =============================================================================
    # 2. FUNCTIONAL LOSSES
    # =============================================================================
    
    print(f"\n2️⃣ Functional Losses")
    print("-" * 25)
    
    def cosine_similarity_loss(predictions, targets):
        """Simple custom loss using cosine similarity."""
        return 1 - nn.functional.cosine_similarity(predictions, targets, dim=-1).mean()
    
    def weighted_mse_loss(predictions, targets):
        """Custom weighted MSE loss."""
        weights = torch.linspace(0.5, 1.5, targets.size(-1)).to(targets.device)
        return (weights * (predictions - targets) ** 2).mean()
    
    functional_losses = {
        "Cosine Similarity": cosine_similarity_loss,
        "Weighted MSE": weighted_mse_loss,
    }
    
    for loss_name, loss_fn in functional_losses.items():
        print(f"\n🔧 Testing {loss_name}...")
        
        model = TabularSSLModel(
            event_encoder=event_encoder,
            sequence_encoder=sequence_encoder,
            corruption=ReConTabCorruption(corruption_rate=0.15),
            custom_loss_fn=loss_fn,  # Custom function with 2 parameters
            learning_rate=1e-3
        )
        
        model.train()
        loss = model._ssl_training_step(x)
        print(f"   ✅ {loss_name}: Loss = {loss:.4f}")
        print(f"   📝 Usage: custom_loss_fn={loss_fn.__name__}")
    
    # =============================================================================
    # 3. COMPLEX SSL LOSSES (Still Supported)
    # =============================================================================
    
    print(f"\n3️⃣ Complex SSL Losses")
    print("-" * 25)
    
    # Custom complex SSL loss
    def custom_ssl_loss(model, representations, original, corrupted_data, ssl_loss_weights):
        """Complex SSL loss with full signature."""
        # Example: Combine reconstruction + contrastive learning
        
        # Reconstruction component
        if not hasattr(model, 'custom_recon_head'):
            repr_dim = representations.size(-1)
            model.custom_recon_head = nn.Linear(repr_dim, original.size(-1)).to(representations.device)
        
        reconstructed = model.custom_recon_head(representations)
        recon_loss = nn.functional.mse_loss(reconstructed, original)
        
        # Contrastive component (simplified)
        norm_repr = nn.functional.normalize(representations, dim=-1)
        batch_size = norm_repr.size(0)
        
        if norm_repr.dim() == 3:  # Handle sequence data
            norm_repr = norm_repr.mean(dim=1)  # Average pooling
        
        sim_matrix = torch.matmul(norm_repr, norm_repr.t())
        contrastive_loss = -torch.log_softmax(sim_matrix, dim=1).diag().mean()
        
        # Combine losses
        recon_weight = ssl_loss_weights.get('reconstruction', 1.0)
        contrastive_weight = ssl_loss_weights.get('contrastive', 0.5)
        
        return recon_weight * recon_loss + contrastive_weight * contrastive_loss
    
    print(f"\n🔧 Testing Custom SSL Loss...")
    
    model = TabularSSLModel(
        event_encoder=event_encoder,
        sequence_encoder=sequence_encoder,
        corruption=SCARFCorruption(corruption_rate=0.6),
        custom_loss_fn=custom_ssl_loss,  # Complex function with full signature
        ssl_loss_weights={'reconstruction': 1.0, 'contrastive': 0.3},
        learning_rate=1e-3
    )
    
    model.train()
    loss = model._ssl_training_step(x)
    print(f"   ✅ Custom SSL Loss: Loss = {loss:.4f}")
    print(f"   📝 Usage: custom_loss_fn=custom_ssl_loss (with full signature)")
    
    # =============================================================================
    # 4. BUILT-IN SSL METHODS (Auto-detection)
    # =============================================================================
    
    print(f"\n4️⃣ Built-in SSL Methods")
    print("-" * 25)
    
    builtin_methods = [
        ("VIME", VIMECorruption(corruption_rate=0.3)),
        ("SCARF", SCARFCorruption(corruption_rate=0.6)),
        ("ReConTab", ReConTabCorruption(corruption_rate=0.15)),
    ]
    
    for method_name, corruption in builtin_methods:
        print(f"\n🔧 Testing {method_name} (auto-detection)...")
        
        model = TabularSSLModel(
            event_encoder=event_encoder,
            sequence_encoder=sequence_encoder,
            projection_head=MLPHead(input_dim=32, output_dim=16) if method_name == "SCARF" else None,
            corruption=corruption,
            # No custom_loss_fn - uses auto-detection!
            learning_rate=1e-3
        )
        
        model.train()
        loss = model._ssl_training_step(x)
        print(f"   ✅ {method_name}: Loss = {loss:.4f}")
        print(f"   📝 Usage: corruption={corruption.__class__.__name__}() (auto-detected)")
    
    # =============================================================================
    # 5. USAGE COMPARISON
    # =============================================================================
    
    print(f"\n📊 Usage Comparison")
    print("=" * 40)
    
    comparison = """
    ┌─────────────────────┬──────────────────────────────────────┬──────────────────┐
    │ Loss Type           │ Usage                                │ Complexity       │
    ├─────────────────────┼──────────────────────────────────────┼──────────────────┤
    │ Simple PyTorch      │ custom_loss_fn=nn.MSELoss()          │ ⭐ (Easiest)    │
    │ Custom Simple       │ custom_loss_fn=my_loss_fn            │ ⭐⭐ (Easy)      │
    │ Built-in SSL        │ corruption=VIMECorruption()          │ ⭐⭐ (Easy)      │
    │ Complex SSL         │ custom_loss_fn=complex_ssl_loss      │ ⭐⭐⭐ (Advanced) │
    └─────────────────────┴──────────────────────────────────────┴──────────────────┘
    
    🎯 Key Benefits:
    ✅ No wrapper functions needed for simple losses
    ✅ Auto-detection handles different function signatures  
    ✅ Backward compatible with complex SSL losses
    ✅ Works with any PyTorch/TorchMetrics loss
    ✅ Automatic tensor dimension handling
    """
    
    print(comparison)
    
    print(f"\n🏆 Success! The system now supports:")
    print(f"   • Simple losses: Just pass nn.MSELoss() directly")
    print(f"   • Custom functions: 2-parameter functions work automatically") 
    print(f"   • SSL methods: Complex 5-parameter functions still supported")
    print(f"   • Auto-detection: Built-in methods work without custom_loss_fn")


def demo_torchmetrics_integration():
    """Demonstrate integration with TorchMetrics (if available)."""
    
    print(f"\n🔍 TorchMetrics Integration Demo")
    print("=" * 35)
    
    try:
        import torchmetrics
        
        print("✅ TorchMetrics available! Testing integration...")
        
        # Create components
        event_encoder = MLPEncoder(input_dim=10, hidden_dims=[32], output_dim=64)
        
        # Test different TorchMetrics losses
        torchmetrics_losses = {
            "MeanSquaredError": torchmetrics.MeanSquaredError(),
            "MeanAbsoluteError": torchmetrics.MeanAbsoluteError(),
        }
        
        x = torch.randn(2, 4, 10)
        
        for metric_name, metric_fn in torchmetrics_losses.items():
            print(f"\n🔧 Testing {metric_name}...")
            
            model = TabularSSLModel(
                event_encoder=event_encoder,
                corruption=VIMECorruption(corruption_rate=0.3),
                custom_loss_fn=metric_fn,  # TorchMetrics work directly!
                learning_rate=1e-3
            )
            
            model.train()
            loss = model._ssl_training_step(x)
            print(f"   ✅ {metric_name}: Loss = {loss:.4f}")
            print(f"   📝 Usage: custom_loss_fn=torchmetrics.{metric_name}()")
        
    except ImportError:
        print("❌ TorchMetrics not available")
        print("   Install with: pip install torchmetrics")
        print("   Then TorchMetrics losses will work directly:")
        print("   custom_loss_fn=torchmetrics.MeanSquaredError()")


if __name__ == "__main__":
    demo_simple_loss_functions()
    demo_torchmetrics_integration() 