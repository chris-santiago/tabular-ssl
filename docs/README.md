# Tabular SSL Documentation

This directory contains the documentation for the Tabular SSL library. The documentation is organized following the Diátaxis framework.

## Structure

- **tutorials/**: Step-by-step guides for beginners
- **how-to-guides/**: Problem-oriented guides for specific tasks
- **reference/**: Technical documentation of the API and components
- **explanation/**: Background information and conceptual explanations

## Building the Documentation

To build the documentation locally:

1. Install the documentation requirements:
```bash
pip install -r docs/requirements.txt
```

2. Build the documentation:
```bash
cd docs
mkdocs build
```

3. Serve the documentation locally:
```bash
mkdocs serve
```

## Documentation Contents

### Key Concepts

- **Modular Components**: Tabular SSL uses a simplified modular architecture with direct component instantiation
- **Configuration System**: Configuration is managed using Hydra with `_target_` specifications for clean component composition
- **Self-Supervised Learning**: Multiple SSL methods are implemented for tabular data
- **Simplified Architecture**: Direct constructor parameters instead of complex configuration classes

### Component Types

The library includes several component types:

- **Event Encoders**: Encode individual events (MLP with flexible architecture)
- **Sequence Encoders**: Encode sequences of events (RNN, LSTM, GRU, Transformer, S4)
- **Embedding Layers**: Handle embedding of categorical features with flexible dimensions
- **Projection Heads**: Project encoded representations to a different space
- **Prediction Heads**: Generate predictions from encoded representations

### Configuration

The configuration system uses Hydra with a structured directory layout:

```
configs/
├── config.yaml                # Main configuration
├── model/                     # Model configurations
│   ├── default.yaml          # Default model assembly
│   ├── event_encoder/        # Event encoder configs
│   ├── sequence_encoder/     # Sequence encoder configs
│   ├── embedding/            # Embedding configs
│   ├── projection_head/      # Projection head configs
│   └── prediction_head/      # Prediction head configs
├── experiments/              # Pre-configured experiments
├── data/                     # Data configurations
├── trainer/                  # Training configurations
├── callbacks/                # Callback configurations
├── logger/                   # Logger configurations
├── paths/                    # Path configurations
├── hydra/                    # Hydra-specific configurations
└── extras/                   # Extra utilities
```

Components are instantiated using Hydra's `_target_` mechanism:

```yaml
# configs/model/event_encoder/mlp.yaml
_target_: tabular_ssl.models.components.MLPEventEncoder
input_dim: 64
hidden_dims: [128, 256]
output_dim: 512
dropout: 0.1
```

## Contributing to the Documentation

Contributions to the documentation are welcome! Please follow these guidelines:

1. Follow the Diátaxis framework organization
2. Maintain a consistent style and tone
3. Include code examples where appropriate
4. Update documentation when making changes to the codebase

## License

The documentation is licensed under the same license as the Tabular SSL library. 