import torch
from tabular_ssl.models.base import ModelConfig
from tabular_ssl.models.example_model import ExampleModel


def create_model_config(model_type: str = "transformer"):
    """Create a sample model configuration with the specified sequence model type."""
    return ModelConfig(
        event_encoder_config={
            "input_dim": 10,
            "hidden_dims": [64, 32],
            "output_dim": 16,
        },
        sequence_encoder_config={
            "model_type": model_type,  # "rnn", "lstm", "gru", "transformer", "ssm", or "s4"
            "input_dim": 16,
            "hidden_dim": 32,
            "num_layers": 2,
            "dropout": 0.1,
            "bidirectional": True,
            # Transformer-specific
            "num_heads": 4,
            # SSM-specific
            "state_dim": 32,
            "use_gate": True,
            # S4-specific
            "max_sequence_length": 1024,
            "use_learnable_dt": True,
            "use_initial_state": True,
        },
        embedding_config={
            "embedding_dims": [
                (5, 8),  # 5 categories, 8-dimensional embedding
                (3, 4),  # 3 categories, 4-dimensional embedding
            ]
        },
        projection_head_config={"input_dim": 32, "hidden_dim": 16, "output_dim": 8},
        prediction_head_config={"input_dim": 8, "num_classes": 2, "dropout": 0.1},
    )


def main():
    # Test different sequence model types
    model_types = ["rnn", "lstm", "gru", "transformer", "ssm", "s4"]

    for model_type in model_types:
        print(f"\nTesting {model_type.upper()} sequence model:")

        # Create model configuration
        config = create_model_config(model_type)

        # Initialize model
        model = ExampleModel(config)

        # Create sample data
        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 10)  # Input features
        y = torch.randint(0, 2, (batch_size,))  # Binary labels

        # Forward pass
        output = model(x)
        print(f"Output shape: {output.shape}")

        # Training step
        loss = model.training_step((x, y), 0)
        print(f"Training loss: {loss.item()}")

        # Validation step
        model.validation_step((x, y), 0)


if __name__ == "__main__":
    main()
