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

- **Component Registry**: Tabular SSL uses a component registry pattern to enable modular and extensible components
- **Configuration System**: Configuration is managed using Hydra, allowing for hierarchical composition
- **Self-Supervised Learning**: Multiple SSL methods are implemented for tabular data
- **Type Safety**: Configuration validation is performed using Pydantic

### Component Types

The library includes several component types:

- **Event Encoders**: Encode individual events (MLP, Autoencoder, Contrastive)
- **Sequence Encoders**: Encode sequences of events (RNN, LSTM, GRU, Transformer, S4)
- **Embedding Layers**: Handle embedding of categorical features
- **Projection Heads**: Project encoded representations to a different space
- **Prediction Heads**: Generate predictions from encoded representations

### Configuration

The configuration system uses Hydra with a structured directory layout:

```
configs/
├── config.yaml                # Main configuration
├── model/                     # Model configurations
│   ├── event_encoder/        # Event encoder configs
│   ├── sequence_encoder/     # Sequence encoder configs
│   ├── embedding/            # Embedding configs
│   ├── projection_head/      # Projection head configs
│   └── prediction_head/      # Prediction head configs
├── data/                     # Data configurations
├── trainer/                  # Training configurations
├── callbacks/                # Callback configurations
├── logger/                   # Logger configurations
├── experiment/               # Experiment configurations
├── hydra/                    # Hydra-specific configurations
└── paths/                    # Path configurations
```

## Contributing to the Documentation

Contributions to the documentation are welcome! Please follow these guidelines:

1. Follow the Diátaxis framework organization
2. Maintain a consistent style and tone
3. Include code examples where appropriate
4. Update documentation when making changes to the codebase

## License

The documentation is licensed under the same license as the Tabular SSL library. 