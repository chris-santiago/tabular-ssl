#!/usr/bin/env python
"""Demonstration of custom loss functions in TabularSSLModel."""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tabular_ssl.models.simplified_base import (
    TabularSSLModel, 
    get_ssl_loss_function,
    vime_loss_fn,
    scarf_loss_fn,
    recontab_loss_fn,
    custom_mixup_loss_fn
)
from tabular_ssl.models.simplified_components import (
    MLPEncoder, TransformerEncoder, MLPHead,
    VIMECorruption, SCARFCorruption, ReConTabCorruption,
    CustomMixupCorruption
)


def custom_contrastive_loss_fn(
    model: TabularSSLModel,
    representations: torch.Tensor,
    original: torch.Tensor,
    corrupted_data: Dict[str, torch.Tensor],
    ssl_loss_weights: Dict[str, float]
) -> torch.Tensor:
    """Example of a completely custom SSL loss function.
    
    This implements a simple contrastive loss that tries to maximize 
    similarity between different corruptions of the same sample.
    """
    batch_size = representations.size(0)
    
    # Create a custom head if needed
    if not hasattr(model, 'custom_contrastive_head'):
        repr_dim = representations.size(-1)
        model.custom_contrastive_head = nn.Sequential(
            nn.Linear(repr_dim, repr_dim // 2),
            nn.ReLU(),
            nn.Linear(repr_dim // 2, 64)  # Project to 64-dim space
        ).to(representations.device)
    
    # Project representations
    projected = model.custom_contrastive_head(representations)
    
    # Normalize for cosine similarity
    projected = F.normalize(projected, dim=1)
    
    # Compute pairwise similarities
    similarity_matrix = torch.matmul(projected, projected.t())
    
    # Create positive pairs (adjacent samples in batch)
    positive_mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)
    for i in range(batch_size - 1):
        positive_mask[i, i + 1] = True
        positive_mask[i + 1, i] = True
    
    # Extract positive and negative similarities
    positives = similarity_matrix[positive_mask]
    negatives = similarity_matrix[~positive_mask & ~torch.eye(batch_size, dtype=torch.bool)]
    
    # Custom contrastive loss: maximize positives, minimize negatives
    positive_loss = -torch.log(torch.sigmoid(positives) + 1e-8).mean()
    negative_loss = -torch.log(1 - torch.sigmoid(negatives) + 1e-8).mean()
    
    contrastive_weight = ssl_loss_weights.get('custom_contrastive', 1.0)
    return contrastive_weight * (positive_loss + negative_loss)


def custom_multi_task_loss_fn(
    model: TabularSSLModel,
    representations: torch.Tensor,
    original: torch.Tensor,
    corrupted_data: Dict[str, torch.Tensor],
    ssl_loss_weights: Dict[str, float]
) -> torch.Tensor:
    """Example of a multi-task custom loss function combining multiple objectives."""
    
    # Task 1: Reconstruction
    if not hasattr(model, 'multi_reconstruction_head'):
        repr_dim = representations.size(-1)
        input_dim = original.size(-1)
        model.multi_reconstruction_head = nn.Linear(repr_dim, input_dim).to(representations.device)
    
    reconstructed = model.multi_reconstruction_head(representations)
    reconstruction_loss = F.mse_loss(reconstructed, original)
    
    # Task 2: Feature prediction (predict mean of features)
    if not hasattr(model, 'feature_predictor'):
        repr_dim = representations.size(-1)
        model.feature_predictor = nn.Linear(repr_dim, 1).to(representations.device)
    
    feature_means = original.mean(dim=-1, keepdim=True)
    predicted_means = model.feature_predictor(representations)
    mean_prediction_loss = F.mse_loss(predicted_means, feature_means)
    
    # Task 3: Variance prediction
    if not hasattr(model, 'variance_predictor'):
        repr_dim = representations.size(-1)
        model.variance_predictor = nn.Linear(repr_dim, 1).to(representations.device)
    
    feature_vars = original.var(dim=-1, keepdim=True)
    predicted_vars = torch.exp(model.variance_predictor(representations))  # Ensure positive
    variance_prediction_loss = F.mse_loss(predicted_vars, feature_vars)
    
    # Combine losses with weights
    recon_weight = ssl_loss_weights.get('reconstruction', 1.0)
    mean_weight = ssl_loss_weights.get('mean_prediction', 0.5)
    var_weight = ssl_loss_weights.get('variance_prediction', 0.3)
    
    total_loss = (recon_weight * reconstruction_loss + 
                  mean_weight * mean_prediction_loss + 
                  var_weight * variance_prediction_loss)
    
    return total_loss


def demonstrate_custom_loss_functions():
    """Demonstrate various ways to use custom loss functions."""
    
    print("🔧 Custom Loss Functions Demonstration")
    print("=" * 50)
    
    # Create sample components
    event_encoder = MLPEncoder(input_dim=10, hidden_dims=[32, 64], output_dim=128)
    sequence_encoder = TransformerEncoder(input_dim=128, hidden_dim=64, num_heads=4, num_layers=2)
    projection_head = MLPHead(input_dim=64, output_dim=32)
    
    # Sample data
    x = torch.randn(4, 8, 10)
    
    print("\n1. Using Built-in Loss Functions")
    print("-" * 35)
    
    # Method 1: Use predefined loss functions
    vime_corruption = VIMECorruption(corruption_rate=0.3)
    
    # Option A: Auto-detection (original way)
    model_auto = TabularSSLModel(
        event_encoder=event_encoder,
        sequence_encoder=sequence_encoder,
        corruption=vime_corruption,
        ssl_loss_weights={'mask_estimation': 1.0, 'value_imputation': 1.0}
    )
    print(f"✅ Auto-detection model: SSL={model_auto.is_ssl}")
    
    # Option B: Explicit loss function using registry
    vime_loss = get_ssl_loss_function('vime')
    model_explicit = TabularSSLModel(
        event_encoder=event_encoder,
        sequence_encoder=sequence_encoder,
        corruption=vime_corruption,
        custom_loss_fn=vime_loss,
        ssl_loss_weights={'mask_estimation': 1.0, 'value_imputation': 1.0}
    )
    print(f"✅ Explicit VIME loss model: SSL={model_explicit.is_ssl}")
    
    print("\n2. Using Completely Custom Loss Functions")
    print("-" * 42)
    
    # Method 2: Custom contrastive loss
    scarf_corruption = SCARFCorruption(corruption_rate=0.6)
    
    model_custom_contrastive = TabularSSLModel(
        event_encoder=event_encoder,
        sequence_encoder=sequence_encoder,
        projection_head=projection_head,
        corruption=scarf_corruption,
        custom_loss_fn=custom_contrastive_loss_fn,
        ssl_loss_weights={'custom_contrastive': 1.5}
    )
    print(f"✅ Custom contrastive model: SSL={model_custom_contrastive.is_ssl}")
    
    # Method 3: Multi-task custom loss
    recontab_corruption = ReConTabCorruption(corruption_rate=0.2)
    
    model_multi_task = TabularSSLModel(
        event_encoder=event_encoder,
        sequence_encoder=sequence_encoder,
        corruption=recontab_corruption,
        custom_loss_fn=custom_multi_task_loss_fn,
        ssl_loss_weights={
            'reconstruction': 1.0,
            'mean_prediction': 0.5,
            'variance_prediction': 0.3
        }
    )
    print(f"✅ Multi-task custom model: SSL={model_multi_task.is_ssl}")
    
    print("\n3. Testing Forward Passes and Loss Computation")
    print("-" * 47)
    
    # Test all models
    models = [
        ("Auto-detection VIME", model_auto),
        ("Explicit VIME", model_explicit),
        ("Custom Contrastive", model_custom_contrastive),
        ("Multi-task Custom", model_multi_task)
    ]
    
    for name, model in models:
        try:
            # Forward pass
            output = model(x)
            print(f"✅ {name}: Forward pass {x.shape} → {output.shape}")
            
            # Training step (simulate)
            model.train()
            loss = model._ssl_training_step(x)
            print(f"   SSL loss: {loss.item():.4f}")
            
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
    
    print("\n4. Configuration Examples")
    print("-" * 25)
    
    # Show how these would work in configuration
    config_examples = {
        "Built-in VIME": {
            "_target_": "tabular_ssl.models.simplified_base.TabularSSLModel",
            "corruption": {"_target_": "tabular_ssl.models.simplified_components.VIMECorruption"},
            # No custom_loss_fn - uses auto-detection
        },
        "Explicit SCARF": {
            "_target_": "tabular_ssl.models.simplified_base.TabularSSLModel",
            "corruption": {"_target_": "tabular_ssl.models.simplified_components.SCARFCorruption"},
            "custom_loss_fn": {"_target_": "tabular_ssl.models.simplified_base.scarf_loss_fn"},
        },
        "Custom Function": {
            "_target_": "tabular_ssl.models.simplified_base.TabularSSLModel",
            "corruption": {"_target_": "your_module.YourCorruption"},
            "custom_loss_fn": {"_target_": "your_module.your_custom_loss_fn"},
            "ssl_loss_weights": {"your_param": 1.0}
        }
    }
    
    for name, config in config_examples.items():
        print(f"📋 {name}:")
        for key, value in config.items():
            if isinstance(value, dict):
                print(f"   {key}: {value.get('_target_', value)}")
            else:
                print(f"   {key}: {value}")
        print()
    
    print("5. Custom Loss Function Benefits")
    print("-" * 33)
    
    benefits = [
        "✅ Use ANY custom SSL objective (contrastive, predictive, generative)",
        "✅ Combine multiple SSL tasks in one loss function",
        "✅ Easy integration with research experiments",
        "✅ Full control over loss computation and model heads",
        "✅ Backward compatibility with auto-detection",
        "✅ Registry system for reusable loss functions",
        "✅ Hydra configuration support for custom functions",
        "✅ Type hints and clear interface for custom functions"
    ]
    
    for benefit in benefits:
        print(f"   {benefit}")
    
    print(f"\n📊 Summary:")
    print(f"   - Auto-detection: Works seamlessly for VIME/SCARF/ReConTab")
    print(f"   - Custom functions: Enable ANY SSL objective") 
    print(f"   - Registry system: Easy access to built-in functions")
    print(f"   - Full extensibility: Research-friendly interface")
    print(f"   - Configuration: Hydra support for custom callables")
    
    print(f"\n🎉 TabularSSLModel now supports arbitrary loss functions!")


if __name__ == "__main__":
    # Import the custom corruption if the extensibility demo exists
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from extensibility_demo import CustomMixupCorruption
    except ImportError:
        # Create a simple custom corruption for demo
        from tabular_ssl.models.simplified_components import BaseCorruption
        
        class CustomMixupCorruption(BaseCorruption):
            def forward(self, x):
                return {
                    'corrupted': x + 0.1 * torch.randn_like(x),
                    'mixup_targets': x,
                    'mixup_lambdas': torch.ones_like(x[:, :1, :1]),
                    'original': x
                }
    
    demonstrate_custom_loss_functions() 