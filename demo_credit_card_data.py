#!/usr/bin/env python3
"""
Demo script for using IBM TabFormer credit card transaction data with Tabular SSL.

This script demonstrates how to:
1. Download the credit card transaction dataset from IBM TabFormer
2. Load and preprocess the data for tabular SSL experiments
3. Train a simple model on the transaction sequences

Usage:
    python demo_credit_card_data.py
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from tabular_ssl.data.sample_data import setup_sample_data, download_credit_card_transactions
from tabular_ssl.data.datamodule import TabularDataModule
import pandas as pd
import numpy as np


def demo_data_download():
    """Demonstrate downloading and exploring the credit card data."""
    print("🏦 Credit Card Transaction Data Demo")
    print("=" * 50)
    
    # Download the data
    print("\n📥 Step 1: Downloading IBM TabFormer credit card data...")
    data_dir = download_credit_card_transactions(data_dir="data", force_download=False)
    print(f"✅ Data available at: {data_dir}")
    
    # Quick setup for exploration
    print("\n🔍 Step 2: Loading and preprocessing sample data...")
    df, metadata = setup_sample_data(
        data_dir="data",
        sequence_length=16,
        n_users=50
    )
    
    # Show data overview
    print(f"\n📊 Dataset Overview:")
    print(f"  • Total transactions: {metadata['n_transactions']}")
    print(f"  • Number of users: {metadata['n_users']}")
    print(f"  • Sequence length: {metadata['sequence_length']}")
    print(f"  • Number of sequences: {metadata['n_sequences']}")
    
    # Show feature information
    print(f"\n🏷️  Features:")
    print(f"  • Categorical features: {len(metadata['categorical_features'])}")
    for feat in metadata['categorical_features'][:3]:  # Show first 3
        print(f"    - {feat['name']}: {feat['num_categories']} categories")
    
    print(f"  • Numerical features: {len(metadata['numerical_features'])}")
    for feat in metadata['numerical_features'][:3]:  # Show first 3
        print(f"    - {feat['name']}: range [{feat['min']:.2f}, {feat['max']:.2f}]")
    
    # Show sample data
    print(f"\n📋 Sample Data (first 5 rows):")
    sample_cols = df.columns[:8] if len(df.columns) > 8 else df.columns
    print(df[sample_cols].head())
    
    return df, metadata


def demo_datamodule():
    """Demonstrate using the DataModule with credit card data."""
    print("\n🔄 Step 3: Testing DataModule integration...")
    
    # Create DataModule with sample data
    datamodule = TabularDataModule(
        data_dir="data",
        use_sample_data=True,
        sample_data_config={
            "data_source": "credit_card",
            "n_users": 100,
            "sequence_length": 32
        },
        sequence_length=32,
        batch_size=16,
        train_val_test_split=[0.7, 0.15, 0.15],
        seed=42
    )
    
    # Prepare and setup data
    print("  📦 Preparing data...")
    datamodule.prepare_data()
    
    print("  ⚙️  Setting up data splits...")
    datamodule.setup(stage="fit")
    
    # Get feature information
    feature_dims = datamodule.feature_processor.get_feature_dims()
    vocab_sizes = datamodule.feature_processor.get_vocab_sizes()
    embedding_dims = datamodule.feature_processor.get_embedding_dims()
    
    print(f"\n📏 Feature Dimensions:")
    print(f"  • Categorical: {feature_dims['categorical']} dims")
    print(f"  • Numerical: {feature_dims['numerical']} dims")
    
    print(f"\n📖 Vocabulary Sizes:")
    for col, size in list(vocab_sizes.items())[:5]:  # Show first 5
        print(f"  • {col}: {size} unique values")
    
    # Test data loaders
    print(f"\n🔄 Data Loaders:")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    
    print(f"  • Train: {len(train_loader)} batches")
    print(f"  • Validation: {len(val_loader)} batches")
    print(f"  • Test: {len(test_loader)} batches")
    
    # Sample a batch
    print(f"\n🎯 Sample Batch:")
    sample_batch = next(iter(train_loader))
    
    for key, tensor in sample_batch.items():
        print(f"  • {key}: {tensor.shape} ({tensor.dtype})")
    
    return datamodule


def demo_training_ready():
    """Demonstrate that the data is ready for training."""
    print("\n🚀 Step 4: Training readiness check...")
    
    # Create a simple model configuration
    model_config = {
        "event_encoder": {
            "input_dim": 64,  # Will be auto-adjusted
            "hidden_dims": [128, 256],
            "output_dim": 512
        },
        "sequence_encoder": {
            "input_dim": 512,
            "hidden_dim": 512,
            "num_layers": 4,
            "num_heads": 8
        }
    }
    
    print("✅ Data is ready for training with:")
    print(f"  • Event encoder: MLP with dims {model_config['event_encoder']['hidden_dims']}")
    print(f"  • Sequence encoder: Transformer with {model_config['sequence_encoder']['num_layers']} layers")
    print(f"  • Sequence length: 32 transactions")
    print(f"  • Batch size: 16 sequences")
    
    print(f"\n💡 To train a model, run:")
    print(f"  python train.py +experiment=credit_card_demo")
    
    print(f"\n📚 Example configurations available:")
    print(f"  • configs/data/credit_card.yaml - Data configuration")
    print(f"  • configs/experiments/credit_card_demo.yaml - Full experiment setup")


def main():
    """Run the complete demo."""
    try:
        # Step 1: Download and explore data
        df, metadata = demo_data_download()
        
        # Step 2: Test DataModule integration
        datamodule = demo_datamodule()
        
        # Step 3: Show training readiness
        demo_training_ready()
        
        print("\n" + "=" * 50)
        print("🎉 Demo completed successfully!")
        print("\n💭 What's next?")
        print("  1. Explore the downloaded data in the 'data/' directory")
        print("  2. Modify configs/experiments/credit_card_demo.yaml for your needs")
        print("  3. Run: python train.py +experiment=credit_card_demo")
        print("  4. Check the results in outputs/ directory")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("💡 Make sure you have all dependencies installed:")
        print("  pip install pandas polars tqdm")
        sys.exit(1)


if __name__ == "__main__":
    main() 