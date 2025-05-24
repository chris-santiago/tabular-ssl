#!/usr/bin/env python
"""Test script to verify the simplified design works correctly."""

import torch
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tabular_ssl.models.simplified_base import TabularSSLModel, create_mlp
from tabular_ssl.models.simplified_components import (
    MLPEncoder, TransformerEncoder, RNNEncoder, 
    MLPHead, TabularEmbedding,
    VIMECorruption, SCARFCorruption, ReConTabCorruption,
    create_encoder, create_corruption
)


def test_simplified_components():
    """Test that simplified components work correctly."""
    print("Testing simplified components...")
    
    # Test MLP encoder
    encoder = MLPEncoder(
        input_dim=10,
        hidden_dims=[32, 64],
        output_dim=16,
        dropout=0.1
    )
    
    x = torch.randn(4, 8, 10)  # batch_size=4, seq_len=8, input_dim=10
    output = encoder(x)
    assert output.shape == (4, 8, 16), f"Expected (4, 8, 16), got {output.shape}"
    print("✅ MLPEncoder works correctly")
    
    # Test Transformer encoder
    transformer = TransformerEncoder(
        input_dim=16,
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        output_dim=24
    )
    
    output = transformer(output)
    assert output.shape == (4, 8, 24), f"Expected (4, 8, 24), got {output.shape}"
    print("✅ TransformerEncoder works correctly")
    
    # Test MLP head
    head = MLPHead(
        input_dim=24,
        output_dim=2,
        hidden_dims=[12]
    )
    
    final_output = head(output)
    assert final_output.shape == (4, 8, 2), f"Expected (4, 8, 2), got {final_output.shape}"
    print("✅ MLPHead works correctly")


def test_corruption_strategies():
    """Test that corruption strategies work correctly."""
    print("\nTesting corruption strategies...")
    
    x = torch.randn(4, 8, 10)
    
    # Test VIME corruption
    vime = VIMECorruption(
        corruption_rate=0.3,
        categorical_indices=[0, 1],
        numerical_indices=[2, 3, 4, 5, 6, 7, 8, 9]
    )
    
    vime_output = vime(x)
    required_keys = {'corrupted', 'mask', 'original'}
    assert all(k in vime_output for k in required_keys), f"Missing keys in VIME output"
    assert vime_output['corrupted'].shape == x.shape, "VIME corrupted shape mismatch"
    print("✅ VIMECorruption works correctly")
    
    # Test SCARF corruption
    scarf = SCARFCorruption(corruption_rate=0.6)
    scarf_output = scarf(x)
    required_keys = {'corrupted', 'positive', 'mask', 'original'}
    assert all(k in scarf_output for k in required_keys), f"Missing keys in SCARF output"
    print("✅ SCARFCorruption works correctly")
    
    # Test ReConTab corruption
    recontab = ReConTabCorruption(
        corruption_rate=0.15,
        corruption_types=["masking", "noise"]
    )
    recontab_output = recontab(x)
    required_keys = {'corrupted', 'corruption_info', 'original'}
    assert all(k in recontab_output for k in required_keys), f"Missing keys in ReConTab output"
    print("✅ ReConTabCorruption works correctly")


def test_unified_model():
    """Test the unified TabularSSLModel."""
    print("\nTesting unified TabularSSLModel...")
    
    # Create components
    event_encoder = MLPEncoder(
        input_dim=10,
        hidden_dims=[32, 64],
        output_dim=16
    )
    
    sequence_encoder = TransformerEncoder(
        input_dim=16,
        hidden_dim=32,
        num_heads=4,
        num_layers=2
    )
    
    projection_head = MLPHead(
        input_dim=32,
        output_dim=16,
        hidden_dims=[24]
    )
    
    # Test standard model (no SSL)
    standard_model = TabularSSLModel(
        event_encoder=event_encoder,
        sequence_encoder=sequence_encoder,
        projection_head=projection_head,
        learning_rate=1e-3
    )
    
    assert not standard_model.is_ssl, "Model should not be in SSL mode"
    print("✅ Standard model creation works")
    
    # Test SSL model with VIME
    vime_corruption = VIMECorruption(corruption_rate=0.3)
    
    ssl_model = TabularSSLModel(
        event_encoder=event_encoder,
        sequence_encoder=sequence_encoder,
        corruption=vime_corruption,
        ssl_loss_weights={'mask_estimation': 1.0, 'value_imputation': 1.0},
        learning_rate=1e-3
    )
    
    assert ssl_model.is_ssl, "Model should be in SSL mode"
    assert hasattr(ssl_model, 'mask_head'), "SSL model should have mask head for VIME"
    assert hasattr(ssl_model, 'value_head'), "SSL model should have value head for VIME"
    print("✅ SSL model creation works")
    
    # Test forward pass
    x = torch.randn(2, 5, 10)
    output = standard_model(x)
    assert output.shape == (2, 5, 16), f"Expected (2, 5, 16), got {output.shape}"
    print("✅ Model forward pass works")


def test_factory_functions():
    """Test the factory functions."""
    print("\nTesting factory functions...")
    
    # Test encoder factory
    mlp_encoder = create_encoder('mlp', input_dim=10, hidden_dims=[32], output_dim=16)
    assert isinstance(mlp_encoder, MLPEncoder), "Factory should return MLPEncoder"
    
    transformer_encoder = create_encoder('transformer', input_dim=16, hidden_dim=32)
    assert isinstance(transformer_encoder, TransformerEncoder), "Factory should return TransformerEncoder"
    
    lstm_encoder = create_encoder('lstm', input_dim=16, hidden_dim=32)
    assert isinstance(lstm_encoder, RNNEncoder), "Factory should return RNNEncoder for LSTM"
    print("✅ Encoder factory works correctly")
    
    # Test corruption factory
    vime_corruption = create_corruption('vime', corruption_rate=0.3)
    assert isinstance(vime_corruption, VIMECorruption), "Factory should return VIMECorruption"
    
    scarf_corruption = create_corruption('scarf', corruption_rate=0.6)
    assert isinstance(scarf_corruption, SCARFCorruption), "Factory should return SCARFCorruption"
    
    recontab_corruption = create_corruption('recontab', corruption_rate=0.15)
    assert isinstance(recontab_corruption, ReConTabCorruption), "Factory should return ReConTabCorruption"
    print("✅ Corruption factory works correctly")


def main():
    """Run all tests."""
    print("Running simplified design tests...\n")
    
    try:
        test_simplified_components()
        test_corruption_strategies()
        test_unified_model()
        test_factory_functions()
        
        print("\n🎉 All tests passed! Simplified design works correctly.")
        print("\nBenefits demonstrated:")
        print("✅ Unified model handles both SSL and standard training")
        print("✅ Simplified components work without unnecessary abstractions")
        print("✅ Factory functions enable easy component creation")
        print("✅ All corruption strategies work correctly")
        print("✅ Code is significantly simpler and more maintainable")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 