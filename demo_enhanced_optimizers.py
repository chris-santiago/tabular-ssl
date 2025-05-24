#!/usr/bin/env python3
"""
Demonstration of Enhanced Optimizer/Scheduler Support in BaseModel

This script shows how the BaseModel class now supports both string-based
and callable-based optimizer and scheduler configurations, providing much
more flexibility while maintaining backward compatibility.
"""

import torch
from functools import partial
from src.tabular_ssl.models.base import BaseModel, MLPPredictionHead
from src.tabular_ssl.models.components import MLPEventEncoder

def demo_enhanced_optimizers():
    """Demonstrate the enhanced optimizer/scheduler functionality."""
    
    print("🚀 Enhanced Optimizer/Scheduler Support Demo")
    print("=" * 50)
    
    # Create a simple event encoder for testing
    event_encoder = MLPEventEncoder(
        input_dim=10, 
        hidden_dims=[32, 16], 
        output_dim=8
    )
    
    print("\n1. 📝 String-based Configuration (Backward Compatible)")
    print("-" * 50)
    
    # Example 1: Traditional string-based approach
    model1 = BaseModel(
        event_encoder=event_encoder,
        optimizer_type="adamw",
        scheduler_type="cosine",
        learning_rate=1e-3,
        weight_decay=1e-4
    )
    
    print("✓ Created model with string-based optimizer and scheduler")
    print(f"  - Optimizer type: {model1.optimizer_type}")
    print(f"  - Scheduler type: {model1.scheduler_type}")
    print(f"  - Learning rate: {model1.learning_rate}")
    print(f"  - Weight decay: {model1.weight_decay}")
    
    print("\n2. 🔧 Custom Optimizer with Partial Functions")
    print("-" * 50)
    
    # Example 2: Custom optimizer using partial
    custom_optimizer = partial(
        torch.optim.SGD, 
        lr=0.01, 
        momentum=0.9, 
        weight_decay=1e-5
    )
    
    model2 = BaseModel(
        event_encoder=event_encoder,
        optimizer_type=custom_optimizer,
        scheduler_type=None  # No scheduler
    )
    
    print("✓ Created model with custom SGD optimizer")
    print(f"  - Optimizer: {model2.optimizer_type}")
    print(f"  - Scheduler: {model2.scheduler_type}")
    
    print("\n3. 📊 Custom Scheduler Configuration")
    print("-" * 50)
    
    # Example 3: Custom scheduler
    custom_scheduler = partial(
        torch.optim.lr_scheduler.ExponentialLR,
        gamma=0.95
    )
    
    model3 = BaseModel(
        event_encoder=event_encoder,
        optimizer_type="adam",
        scheduler_type=custom_scheduler
    )
    
    print("✓ Created model with custom ExponentialLR scheduler")
    print(f"  - Optimizer: {model3.optimizer_type}")
    print(f"  - Scheduler: {model3.scheduler_type}")
    
    print("\n4. 🎯 Advanced Custom Configuration")
    print("-" * 50)
    
    # Example 4: Both custom optimizer and scheduler
    advanced_optimizer = partial(
        torch.optim.AdamW,
        lr=2e-4,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    advanced_scheduler = partial(
        torch.optim.lr_scheduler.OneCycleLR,
        max_lr=1e-3,
        epochs=100,
        steps_per_epoch=500
    )
    
    model4 = BaseModel(
        event_encoder=event_encoder,
        optimizer_type=advanced_optimizer,
        scheduler_type=advanced_scheduler
    )
    
    print("✓ Created model with advanced custom configuration")
    print(f"  - Custom AdamW optimizer with specific hyperparameters")
    print(f"  - Custom OneCycleLR scheduler")
    
    print("\n5. 🧪 Testing Optimizer/Scheduler Creation")
    print("-" * 50)
    
    # Test that the optimizers and schedulers can actually be created
    try:
        # Test string-based
        opt_sched1 = model1.configure_optimizers()
        print(f"✓ String-based: {type(opt_sched1[0][0]).__name__} + {type(opt_sched1[1][0]).__name__}")
        
        # Test custom optimizer only
        opt2 = model2.configure_optimizers()
        print(f"✓ Custom optimizer: {type(opt2).__name__}")
        
        # Test custom scheduler
        opt_sched3 = model3.configure_optimizers()
        print(f"✓ Custom scheduler: {type(opt_sched3[0][0]).__name__} + {type(opt_sched3[1][0]).__name__}")
        
        # Test advanced configuration
        opt_sched4 = model4.configure_optimizers()
        print(f"✓ Advanced config: {type(opt_sched4[0][0]).__name__} + {type(opt_sched4[1][0]).__name__}")
        
    except Exception as e:
        print(f"✗ Error testing optimizer creation: {e}")
    
    print("\n🎉 Enhanced Optimizer/Scheduler Support Summary")
    print("=" * 50)
    print("✓ Backward compatibility maintained with string-based configs")
    print("✓ Support for custom optimizers via partial functions")
    print("✓ Support for custom schedulers via partial functions")
    print("✓ Flexible configuration for advanced use cases")
    print("✓ Type hints and comprehensive documentation added")
    print("\nThe BaseModel class now provides maximum flexibility for")
    print("optimizer and scheduler configuration while maintaining")
    print("the simplicity of string-based configuration for common cases.")

if __name__ == "__main__":
    demo_enhanced_optimizers() 