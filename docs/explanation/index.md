# Explanation

This section provides background information and explanations of the concepts and design decisions behind Tabular SSL.

## Available Topics

- [Architecture Overview](architecture.md) - System design and components
- [SSL Methods](ssl-methods.md) - Self-supervised learning approaches
- [Performance Considerations](performance.md) - Optimization and scaling

## Key Concepts

### Self-Supervised Learning
Self-supervised learning (SSL) is a machine learning paradigm where models learn from unlabeled data by creating their own supervision signals. In the context of tabular data, this involves:

- Feature masking and reconstruction
- Contrastive learning
- Predictive tasks

### Architecture
The Tabular SSL architecture is designed to:

- Handle mixed data types (numerical and categorical)
- Process variable-length sequences
- Learn robust representations
- Scale to large datasets

### Performance
Key performance considerations include:

- Memory efficiency
- Training speed
- Model complexity
- Inference latency

## Related Resources

- [Tutorials](../tutorials/index.md) - Step-by-step guides
- [How-to Guides](../how-to-guides/index.md) - Practical solutions
- [Reference](../reference/index.md) - Technical documentation 