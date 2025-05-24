#!/usr/bin/env python3
"""
Demo script for VIME, SCARF, and ReConTab corruption strategies.

This script demonstrates the different corruption strategies implemented
for tabular self-supervised learning, showing how each method corrupts
data and what outputs they produce.

Usage:
    python demo_corruption_strategies.py
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from tabular_ssl.models.components import VIMECorruption, SCARFCorruption, ReConTabCorruption
from tabular_ssl.data.sample_data import setup_sample_data


def create_sample_data():
    """Create sample tabular data for demonstration."""
    print("📊 Creating sample tabular data...")
    
    # Create synthetic mixed tabular data
    batch_size, seq_len, num_features = 8, 16, 10
    
    # Create mixed categorical and numerical features
    categorical_features = torch.randint(0, 5, (batch_size, seq_len, 3))  # 3 categorical features
    numerical_features = torch.randn(batch_size, seq_len, 7) * 2 + 1      # 7 numerical features
    
    # Combine features
    data = torch.cat([categorical_features.float(), numerical_features], dim=-1)
    
    print(f"  • Data shape: {data.shape}")
    print(f"  • Categorical features: indices 0-2 (values 0-4)")
    print(f"  • Numerical features: indices 3-9 (normal distribution)")
    
    return data, list(range(3)), list(range(3, 10))


def demo_vime_corruption():
    """Demonstrate VIME corruption strategy."""
    print("\n🎭 VIME Corruption Strategy Demo")
    print("=" * 50)
    
    data, categorical_indices, numerical_indices = create_sample_data()
    
    # Initialize VIME corruption
    vime = VIMECorruption(
        corruption_rate=0.3,
        categorical_indices=categorical_indices,
        numerical_indices=numerical_indices
    )
    
    # Set feature distributions from data
    vime.set_feature_distributions(data, categorical_indices, numerical_indices)
    
    print(f"📋 VIME Configuration:")
    print(f"  • Corruption rate: 30%")
    print(f"  • Categorical vocab sizes: {vime.categorical_vocab_sizes}")
    print(f"  • Numerical distributions (first 3): {dict(list(vime.numerical_distributions.items())[:3])}")
    
    # Apply corruption
    vime.train()  # Set to training mode
    corrupted_data, mask = vime(data)
    
    # Show results
    print(f"\n📊 Corruption Results:")
    print(f"  • Original data shape: {data.shape}")
    print(f"  • Corrupted data shape: {corrupted_data.shape}")
    print(f"  • Mask shape: {mask.shape}")
    print(f"  • Corruption rate achieved: {mask.mean().item():.3f}")
    
    # Show examples
    print(f"\n🔍 Sample Corruptions (first sample, first sequence position):")
    sample_idx, seq_idx = 0, 0
    for feat_idx in range(min(5, data.shape[-1])):
        original = data[sample_idx, seq_idx, feat_idx].item()
        corrupted = corrupted_data[sample_idx, seq_idx, feat_idx].item()
        is_masked = mask[sample_idx, seq_idx, feat_idx].item()
        
        feat_type = "categorical" if feat_idx in categorical_indices else "numerical"
        print(f"  • Feature {feat_idx} ({feat_type}): {original:.3f} → {corrupted:.3f} (masked: {bool(is_masked)})")
    
    return data, corrupted_data, mask


def demo_scarf_corruption():
    """Demonstrate SCARF corruption strategy."""
    print("\n🌟 SCARF Corruption Strategy Demo")
    print("=" * 50)
    
    data, _, _ = create_sample_data()
    
    # Initialize SCARF corruption
    scarf = SCARFCorruption(
        corruption_rate=0.6,
        corruption_strategy="random_swap"
    )
    
    print(f"📋 SCARF Configuration:")
    print(f"  • Corruption rate: 60% of features")
    print(f"  • Strategy: Random feature swapping")
    
    # Apply corruption
    scarf.train()  # Set to training mode
    corrupted_data = scarf(data)
    
    # Create contrastive pairs
    view1, view2 = scarf.create_contrastive_pairs(data)
    
    print(f"\n📊 Corruption Results:")
    print(f"  • Original data shape: {data.shape}")
    print(f"  • Single corrupted view shape: {corrupted_data.shape}")
    print(f"  • Contrastive view 1 shape: {view1.shape}")
    print(f"  • Contrastive view 2 shape: {view2.shape}")
    
    # Measure feature-wise corruption
    feature_corruption_rates = []
    for feat_idx in range(data.shape[-1]):
        original_feat = data[:, :, feat_idx]
        corrupted_feat = corrupted_data[:, :, feat_idx]
        corruption_rate = (original_feat != corrupted_feat).float().mean().item()
        feature_corruption_rates.append(corruption_rate)
    
    avg_corruption_rate = np.mean(feature_corruption_rates)
    print(f"  • Average feature corruption rate: {avg_corruption_rate:.3f}")
    
    # Show examples
    print(f"\n🔍 Sample Corruptions (first sample, first sequence position):")
    sample_idx, seq_idx = 0, 0
    for feat_idx in range(min(5, data.shape[-1])):
        original = data[sample_idx, seq_idx, feat_idx].item()
        corrupted = corrupted_data[sample_idx, seq_idx, feat_idx].item()
        is_corrupted = original != corrupted
        
        print(f"  • Feature {feat_idx}: {original:.3f} → {corrupted:.3f} (corrupted: {is_corrupted})")
    
    return data, corrupted_data, view1, view2


def demo_recontab_corruption():
    """Demonstrate ReConTab corruption strategy."""
    print("\n🔧 ReConTab Corruption Strategy Demo")
    print("=" * 50)
    
    data, categorical_indices, numerical_indices = create_sample_data()
    
    # Initialize ReConTab corruption
    recontab = ReConTabCorruption(
        corruption_rate=0.15,
        categorical_indices=categorical_indices,
        numerical_indices=numerical_indices,
        corruption_types=["masking", "noise", "swapping"],
        masking_strategy="random",
        noise_std=0.1,
        swap_probability=0.1
    )
    
    print(f"📋 ReConTab Configuration:")
    print(f"  • Base corruption rate: 15%")
    print(f"  • Corruption types: {recontab.corruption_types}")
    print(f"  • Masking strategy: {recontab.masking_strategy}")
    print(f"  • Noise std: {recontab.noise_std}")
    
    # Apply corruption
    recontab.train()  # Set to training mode
    corrupted_data, corruption_info = recontab(data)
    
    # Get reconstruction targets
    targets = recontab.reconstruction_targets(data, corrupted_data, corruption_info)
    
    print(f"\n📊 Corruption Results:")
    print(f"  • Original data shape: {data.shape}")
    print(f"  • Corrupted data shape: {corrupted_data.shape}")
    print(f"  • Corruption info shape: {corruption_info.shape}")
    
    # Analyze corruption types
    corruption_stats = {}
    for corruption_type, type_id in [("original", 0), ("masked", 1), ("noise", 2), ("swapped", 3)]:
        count = (corruption_info == type_id).sum().item()
        total = corruption_info.numel()
        corruption_stats[corruption_type] = count / total
    
    print(f"\n📈 Corruption Type Distribution:")
    for corruption_type, ratio in corruption_stats.items():
        print(f"  • {corruption_type.title()}: {ratio:.3f} ({ratio*100:.1f}%)")
    
    print(f"\n🎯 Reconstruction Targets Available:")
    for target_name, target_data in targets.items():
        if isinstance(target_data, torch.Tensor):
            print(f"  • {target_name}: {target_data.shape}")
        else:
            print(f"  • {target_name}: {type(target_data)}")
    
    # Show examples
    print(f"\n🔍 Sample Corruptions (first sample, first sequence position):")
    sample_idx, seq_idx = 0, 0
    for feat_idx in range(min(5, data.shape[-1])):
        original = data[sample_idx, seq_idx, feat_idx].item()
        corrupted = corrupted_data[sample_idx, seq_idx, feat_idx].item()
        corruption_type_id = corruption_info[sample_idx, seq_idx, feat_idx].item()
        
        corruption_types = {0: "original", 1: "masked", 2: "noise", 3: "swapped"}
        corruption_type = corruption_types.get(int(corruption_type_id), "unknown")
        
        print(f"  • Feature {feat_idx}: {original:.3f} → {corrupted:.3f} (type: {corruption_type})")
    
    return data, corrupted_data, corruption_info, targets


def demo_corruption_comparison():
    """Compare all three corruption strategies on the same data."""
    print("\n⚖️  Corruption Strategy Comparison")
    print("=" * 50)
    
    data, categorical_indices, numerical_indices = create_sample_data()
    
    # Initialize all corruption strategies
    vime = VIMECorruption(corruption_rate=0.3, categorical_indices=categorical_indices, numerical_indices=numerical_indices)
    vime.set_feature_distributions(data, categorical_indices, numerical_indices)
    
    scarf = SCARFCorruption(corruption_rate=0.6, corruption_strategy="random_swap")
    
    recontab = ReConTabCorruption(
        corruption_rate=0.15, 
        categorical_indices=categorical_indices, 
        numerical_indices=numerical_indices
    )
    
    # Set all to training mode
    vime.train()
    scarf.train()
    recontab.train()
    
    # Apply all corruptions
    vime_corrupted, vime_mask = vime(data)
    scarf_corrupted = scarf(data)
    recontab_corrupted, recontab_info = recontab(data)
    
    print(f"📊 Comparison Results:")
    print(f"  • Original data: {data.shape}")
    
    # VIME analysis
    vime_rate = vime_mask.mean().item()
    print(f"  • VIME corruption rate: {vime_rate:.3f}")
    
    # SCARF analysis
    scarf_differences = (data != scarf_corrupted).float().mean().item()
    print(f"  • SCARF difference rate: {scarf_differences:.3f}")
    
    # ReConTab analysis
    recontab_corrupted_positions = (recontab_info > 0).float().mean().item()
    print(f"  • ReConTab corruption rate: {recontab_corrupted_positions:.3f}")
    
    print(f"\n🎯 Key Differences:")
    print(f"  • VIME: Returns corruption mask for mask estimation task")
    print(f"  • SCARF: Designed for contrastive learning, high corruption rate")
    print(f"  • ReConTab: Multiple corruption types with detailed corruption info")
    
    print(f"\n📝 Use Cases:")
    print(f"  • VIME: Value imputation + mask estimation pretext tasks")
    print(f"  • SCARF: Contrastive representation learning")
    print(f"  • ReConTab: Multi-task reconstruction with contrastive learning")


def main():
    """Run the complete corruption strategies demo."""
    try:
        print("🎭 Tabular SSL Corruption Strategies Demo")
        print("=" * 60)
        
        # Demo each corruption strategy
        demo_vime_corruption()
        demo_scarf_corruption()
        demo_recontab_corruption()
        
        # Compare strategies
        demo_corruption_comparison()
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("\n💭 What's next?")
        print("  1. Try these corruption strategies with real data:")
        print("     python train.py +experiment=vime_ssl")
        print("     python train.py +experiment=scarf_ssl")
        print("     python train.py +experiment=recontab_ssl")
        print("  2. Experiment with different corruption parameters")
        print("  3. Combine corruption strategies for hybrid approaches")
        print("  4. Evaluate on downstream tasks")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("💡 Make sure you have PyTorch installed:")
        print("  pip install torch")
        sys.exit(1)


if __name__ == "__main__":
    main() 