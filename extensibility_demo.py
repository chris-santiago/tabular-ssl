#!/usr/bin/env python
"""Demonstration of extensibility in the simplified design."""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tabular_ssl.models.simplified_base import TabularSSLModel
from tabular_ssl.models.simplified_components import (
    BaseCorruption, MLPHead, create_encoder, create_corruption
)


# =============================================================================
# CUSTOM ENCODERS - Easy to add new architectures
# =============================================================================

class CustomConvEncoder(nn.Module):
    """Custom 1D convolutional encoder for tabular sequences."""
    
    def __init__(
        self,
        input_dim: int,
        num_filters: int = 64,
        kernel_sizes: List[int] = [3, 5, 7],
        output_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Multiple conv layers with different kernel sizes
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_dim, num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        # Combine multiple conv outputs
        total_filters = num_filters * len(kernel_sizes)
        self.projection = nn.Sequential(
            nn.Linear(total_filters, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, seq_len, features) -> (batch, seq_len, output_dim)"""
        # Conv1d expects (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # Apply multiple convolutions
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = F.relu(conv(x))  # (batch, num_filters, seq_len)
            conv_outputs.append(conv_out)
        
        # Concatenate along feature dimension
        combined = torch.cat(conv_outputs, dim=1)  # (batch, total_filters, seq_len)
        
        # Back to (batch, seq_len, total_filters)
        combined = combined.transpose(1, 2)
        
        # Project to output dimension
        output = self.projection(combined)
        return output


class CustomAttentionEncoder(nn.Module):
    """Custom attention-based encoder (different from standard Transformer)."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        output_dim: Optional[int] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or hidden_dim
        
        # Custom attention mechanism
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)
        
        self.multihead_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
        self.output_proj = nn.Linear(hidden_dim, self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Custom attention forward pass."""
        # Project to hidden space
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)
        
        # Apply attention
        attn_output, _ = self.multihead_attn(query, key, value)
        
        # Project to output dimension and normalize
        output = self.output_proj(attn_output)
        output = self.norm(output)
        
        return output


# =============================================================================
# CUSTOM CORRUPTION STRATEGIES - Easy to add new SSL methods
# =============================================================================

class CustomMixupCorruption(BaseCorruption):
    """Custom Mixup-based corruption for tabular SSL."""
    
    def __init__(
        self,
        corruption_rate: float = 0.5,
        alpha: float = 0.2,  # Mixup alpha parameter
        **kwargs
    ):
        super().__init__(corruption_rate, **kwargs)
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply Mixup corruption between samples."""
        batch_size, seq_len, feature_dim = x.shape
        
        # Generate mixup coefficients
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample((batch_size, 1, 1))
        else:
            lam = torch.ones(batch_size, 1, 1)
        
        lam = lam.to(x.device)
        
        # Random permutation for mixing
        batch_indices = torch.randperm(batch_size, device=x.device)
        
        # Apply corruption mask
        corruption_mask = torch.rand(batch_size, seq_len, feature_dim, device=x.device) < self.corruption_rate
        
        # Mixup corruption
        x_mixed = x.clone()
        x_permuted = x[batch_indices]
        
        x_mixed = torch.where(
            corruption_mask,
            lam * x + (1 - lam) * x_permuted,
            x
        )
        
        return {
            'corrupted': x_mixed,
            'mixup_targets': x_permuted,
            'mixup_lambdas': lam,
            'corruption_mask': corruption_mask,
            'original': x
        }


class CustomCutMixCorruption(BaseCorruption):
    """Custom CutMix-based corruption for sequences."""
    
    def __init__(
        self,
        corruption_rate: float = 0.3,
        beta: float = 1.0,  # CutMix beta parameter
        **kwargs
    ):
        super().__init__(corruption_rate, **kwargs)
        self.beta = beta
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply CutMix corruption to sequence segments."""
        batch_size, seq_len, feature_dim = x.shape
        
        # Generate cut ratios
        if self.beta > 0:
            cut_ratio = torch.distributions.Beta(self.beta, self.beta).sample((batch_size,))
        else:
            cut_ratio = torch.ones(batch_size) * 0.5
        
        cut_ratio = cut_ratio.to(x.device)
        
        # Random permutation
        batch_indices = torch.randperm(batch_size, device=x.device)
        
        corrupted = x.clone()
        cut_masks = torch.zeros_like(x, dtype=torch.bool)
        
        for i in range(batch_size):
            if torch.rand(1).item() < self.corruption_rate:
                # Calculate cut size
                cut_size = int(seq_len * cut_ratio[i])
                if cut_size > 0:
                    # Random cut position
                    start_pos = torch.randint(0, seq_len - cut_size + 1, (1,)).item()
                    end_pos = start_pos + cut_size
                    
                    # Apply cut
                    cut_masks[i, start_pos:end_pos, :] = True
                    corrupted[i, start_pos:end_pos, :] = x[batch_indices[i], start_pos:end_pos, :]
        
        return {
            'corrupted': corrupted,
            'cut_masks': cut_masks,
            'cut_ratios': cut_ratio,
            'original': x
        }


# =============================================================================
# EXTEND THE SIMPLIFIED DESIGN
# =============================================================================

def extend_factories():
    """Show how to extend factory functions with custom components."""
    
    # Extend encoder factory
    def create_extended_encoder(encoder_type: str, **kwargs) -> nn.Module:
        """Extended encoder factory with custom encoders."""
        
        # Use original factory for built-in types
        original_map = {
            'mlp': 'tabular_ssl.models.simplified_components.MLPEncoder',
            'transformer': 'tabular_ssl.models.simplified_components.TransformerEncoder',
            'rnn': 'tabular_ssl.models.simplified_components.RNNEncoder',
        }
        
        # Add custom encoders
        custom_map = {
            'conv': CustomConvEncoder,
            'custom_attention': CustomAttentionEncoder,
            'multi_conv': lambda **k: CustomConvEncoder(kernel_sizes=[3, 5, 7, 9], **k),
        }
        
        if encoder_type in custom_map:
            return custom_map[encoder_type](**kwargs)
        elif encoder_type in original_map:
            # Use original factory
            return create_encoder(encoder_type, **kwargs)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    # Extend corruption factory
    def create_extended_corruption(corruption_type: str, **kwargs) -> BaseCorruption:
        """Extended corruption factory with custom corruptions."""
        
        # Add custom corruptions
        custom_map = {
            'mixup': CustomMixupCorruption,
            'cutmix': CustomCutMixCorruption,
            'mixup_strong': lambda **k: CustomMixupCorruption(alpha=0.5, **k),
        }
        
        if corruption_type in custom_map:
            return custom_map[corruption_type](**kwargs)
        else:
            # Use original factory
            return create_corruption(corruption_type, **kwargs)
    
    return create_extended_encoder, create_extended_corruption


def extend_ssl_model():
    """Show how to extend TabularSSLModel with custom SSL logic."""
    
    class ExtendedTabularSSLModel(TabularSSLModel):
        """Extended model with custom SSL loss computation."""
        
        def _ssl_training_step(self, batch):
            """Extended SSL training with custom corruption support."""
            x = batch['features'] if isinstance(batch, dict) else batch
            
            # Apply corruption
            corrupted_data = self.corruption(x)
            x_corrupted = corrupted_data['corrupted']
            
            # Get representations from corrupted data
            representations = self.encode(x_corrupted)
            
            # Extended corruption detection
            corruption_name = self.corruption.__class__.__name__.lower()
            
            if "vime" in corruption_name:
                loss = self._compute_vime_loss(representations, x, corrupted_data)
            elif "scarf" in corruption_name:
                loss = self._compute_scarf_loss(representations, x, corrupted_data)
            elif "recontab" in corruption_name:
                loss = self._compute_recontab_loss(representations, x, corrupted_data)
            elif "mixup" in corruption_name:
                loss = self._compute_mixup_loss(representations, x, corrupted_data)
            elif "cutmix" in corruption_name:
                loss = self._compute_cutmix_loss(representations, x, corrupted_data)
            else:
                # Generic reconstruction loss for unknown corruptions
                if hasattr(self, 'reconstruction_head'):
                    reconstructed = self.reconstruction_head(representations)
                    loss = F.mse_loss(reconstructed, x)
                else:
                    # Create generic reconstruction head on-the-fly
                    repr_dim = representations.size(-1)
                    input_dim = x.size(-1)
                    self.reconstruction_head = nn.Linear(repr_dim, input_dim).to(x.device)
                    reconstructed = self.reconstruction_head(representations)
                    loss = F.mse_loss(reconstructed, x)
                    print(f"⚠️  Unknown corruption type '{corruption_name}', using generic reconstruction loss")
            
            self.log('train/ssl_loss', loss, on_step=True, on_epoch=True)
            return loss
        
        def _compute_mixup_loss(self, representations, original, corrupted_data):
            """Compute Mixup-based SSL loss."""
            if not hasattr(self, 'mixup_head'):
                repr_dim = representations.size(-1)
                self.mixup_head = nn.Linear(repr_dim, repr_dim).to(representations.device)
            
            # Predict mixed representations
            pred_mixed = self.mixup_head(representations)
            
            # Get mixup targets and lambdas
            mixup_targets = corrupted_data['mixup_targets']
            mixup_lambdas = corrupted_data['mixup_lambdas']
            
            # Mixup loss: predict the mixing coefficient
            target_mixed = mixup_lambdas * original + (1 - mixup_lambdas) * mixup_targets
            
            mixup_loss = F.mse_loss(pred_mixed, target_mixed)
            
            # Add lambda prediction loss
            lambda_weight = self.ssl_loss_weights.get('mixup_lambda', 0.1)
            if lambda_weight > 0:
                lambda_pred = torch.sigmoid(self.mixup_head(representations).mean(dim=-1, keepdim=True))
                lambda_loss = F.mse_loss(lambda_pred, mixup_lambdas)
                mixup_loss += lambda_weight * lambda_loss
            
            return mixup_loss
        
        def _compute_cutmix_loss(self, representations, original, corrupted_data):
            """Compute CutMix-based SSL loss."""
            if not hasattr(self, 'cutmix_head'):
                repr_dim = representations.size(-1)
                self.cutmix_head = nn.Linear(repr_dim, 1).to(representations.device)
            
            # Predict cut masks
            cut_pred = torch.sigmoid(self.cutmix_head(representations))
            cut_masks = corrupted_data['cut_masks'].float()
            
            # Binary classification loss for cut regions
            cut_loss = F.binary_cross_entropy(cut_pred.squeeze(-1), cut_masks.mean(dim=-1))
            
            return cut_loss
    
    return ExtendedTabularSSLModel


def demonstrate_extensibility():
    """Demonstrate the extensibility of the simplified design."""
    
    print("🔧 Extensibility Demonstration: Custom Components")
    print("=" * 60)
    
    # 1. Custom Encoders
    print("\n1. Custom Encoders")
    print("-" * 20)
    
    try:
        # Test custom convolutional encoder
        conv_encoder = CustomConvEncoder(
            input_dim=10,
            num_filters=32,
            kernel_sizes=[3, 5],
            output_dim=64
        )
        
        x = torch.randn(4, 8, 10)
        conv_output = conv_encoder(x)
        print(f"✅ CustomConvEncoder: {x.shape} → {conv_output.shape}")
        
        # Test custom attention encoder
        attn_encoder = CustomAttentionEncoder(
            input_dim=64,  # Match conv_encoder output_dim
            hidden_dim=32,
            num_heads=4,
            output_dim=48
        )
        
        attn_output = attn_encoder(conv_output)  # Use conv_output instead of x
        print(f"✅ CustomAttentionEncoder: {conv_output.shape} → {attn_output.shape}")
        
    except Exception as e:
        print(f"❌ Error testing custom encoders: {e}")
        return
    
    # 2. Custom Corruptions
    print("\n2. Custom Corruption Strategies")
    print("-" * 32)
    
    try:
        # Test Mixup corruption
        mixup_corruption = CustomMixupCorruption(
            corruption_rate=0.5,
            alpha=0.2
        )
        
        mixup_output = mixup_corruption(x)
        expected_keys = {'corrupted', 'mixup_targets', 'mixup_lambdas', 'corruption_mask', 'original'}
        assert all(k in mixup_output for k in expected_keys)
        print(f"✅ CustomMixupCorruption: {list(mixup_output.keys())}")
        
        # Test CutMix corruption
        cutmix_corruption = CustomCutMixCorruption(
            corruption_rate=0.3,
            beta=1.0
        )
        
        cutmix_output = cutmix_corruption(x)
        expected_keys = {'corrupted', 'cut_masks', 'cut_ratios', 'original'}
        assert all(k in cutmix_output for k in expected_keys)
        print(f"✅ CustomCutMixCorruption: {list(cutmix_output.keys())}")
        
    except Exception as e:
        print(f"❌ Error testing custom corruptions: {e}")
        return
    
    # 3. Extended Factories
    print("\n3. Extended Factory Functions")
    print("-" * 30)
    
    try:
        create_extended_encoder, create_extended_corruption = extend_factories()
        
        # Test extended encoder factory
        custom_encoders = ['conv', 'custom_attention', 'multi_conv']
        for enc_type in custom_encoders:
            encoder = create_extended_encoder(enc_type, input_dim=10, output_dim=32)
            print(f"✅ Factory created {enc_type}: {type(encoder).__name__}")
        
        # Test extended corruption factory
        custom_corruptions = ['mixup', 'cutmix', 'mixup_strong']
        for corr_type in custom_corruptions:
            corruption = create_extended_corruption(corr_type)
            print(f"✅ Factory created {corr_type}: {type(corruption).__name__}")
        
    except Exception as e:
        print(f"❌ Error testing extended factories: {e}")
        return
    
    # 4. Extended SSL Model
    print("\n4. Extended SSL Model with Custom Loss")
    print("-" * 38)
    
    try:
        ExtendedTabularSSLModel = extend_ssl_model()
        
        # Create model with custom components
        model = ExtendedTabularSSLModel(
            event_encoder=conv_encoder,
            sequence_encoder=attn_encoder,
            projection_head=MLPHead(input_dim=48, output_dim=32),  # Match attn_encoder output
            corruption=mixup_corruption,
            ssl_loss_weights={
                'mixup_lambda': 0.1
            }
        )
        
        print(f"✅ Created ExtendedTabularSSLModel")
        print(f"   - Event encoder: {type(model.event_encoder).__name__}")
        print(f"   - Sequence encoder: {type(model.sequence_encoder).__name__}")
        print(f"   - Corruption: {type(model.corruption).__name__}")
        print(f"   - SSL mode: {model.is_ssl}")
        
        # Test forward pass
        output = model(x)
        print(f"✅ Forward pass: {x.shape} → {output.shape}")
        
    except Exception as e:
        print(f"❌ Error testing extended SSL model: {e}")
        return
    
    # 5. Configuration Extensibility
    print("\n5. Configuration Extensibility")
    print("-" * 30)
    
    custom_config_example = {
        "_target_": "ExtendedTabularSSLModel",  # Your custom model class
        "event_encoder": {
            "_target_": "CustomConvEncoder",  # Your custom encoder
            "input_dim": 10,
            "num_filters": 64,
            "kernel_sizes": [3, 5, 7],
            "output_dim": 128
        },
        "sequence_encoder": {
            "_target_": "CustomAttentionEncoder",  # Your custom sequence encoder
            "input_dim": 128,
            "hidden_dim": 256,
            "num_heads": 8
        },
        "corruption": {
            "_target_": "CustomMixupCorruption",  # Your custom corruption
            "corruption_rate": 0.5,
            "alpha": 0.2
        },
        "ssl_loss_weights": {
            "mixup_lambda": 0.1  # Custom SSL parameters
        }
    }
    
    print("📋 Example custom configuration:")
    for key, value in custom_config_example.items():
        if isinstance(value, dict) and "_target_" in value:
            print(f"   {key}: {value['_target_']}")
        else:
            print(f"   {key}: {value}")
    
    # 6. Extensibility Benefits
    print("\n6. Extensibility Benefits")
    print("-" * 25)
    
    benefits = [
        "✅ Any nn.Module can be used as encoder/corruption",
        "✅ Factory functions are easily extensible",
        "✅ Custom SSL loss methods can be added",
        "✅ Hydra configs work with any _target_ class",
        "✅ No inheritance required for custom components",
        "✅ Name-based detection for automatic SSL head creation",
        "✅ Generic fallback for unknown corruption types",
        "✅ Unified parameter structure for all SSL methods"
    ]
    
    for benefit in benefits:
        print(f"   {benefit}")
    
    print(f"\n📊 Summary:")
    print(f"   - Encoders: Use ANY nn.Module (conv, attention, custom, etc.)")
    print(f"   - Corruptions: Use ANY corruption strategy (mixup, cutmix, custom, etc.)")
    print(f"   - SSL Detection: Automatic via naming + manual fallback")
    print(f"   - Configuration: Full Hydra support for any custom class")
    print(f"   - Extension: Simple inheritance or composition patterns")
    
    print(f"\n🎉 Simplified design is MORE extensible than the original!")


if __name__ == "__main__":
    demonstrate_extensibility() 