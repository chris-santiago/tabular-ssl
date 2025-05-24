# Tabular SSL Documentation

Welcome to the documentation for **Tabular SSL** - a modular library for self-supervised learning on tabular data. This documentation follows the [Diátaxis framework](https://diataxis.fr/) for clear, purpose-driven content organization.

## Documentation Structure

### 📚 [Tutorials](tutorials/)
**Learning-oriented guides for newcomers**

Step-by-step lessons to learn Tabular SSL fundamentals:
- [Getting Started](tutorials/getting-started.md) - Install and run your first experiment (10 min)
- [Basic Usage](tutorials/basic-usage.md) - Core concepts and workflows
- [Custom Components](tutorials/custom-components.md) - Create your own components (20 min)

### 🛠️ [How-to Guides](how-to-guides/)
**Problem-oriented solutions for specific tasks**

Practical solutions to common challenges:
- [Model Training](how-to-guides/model-training.md) - Solve training problems
- [Data Preparation](how-to-guides/data-preparation.md) - Handle your datasets
- [Evaluation](how-to-guides/evaluation.md) - Evaluate model performance
- [Configuring Experiments](how-to-guides/configuring-experiments.md) - Set up custom experiments

### 📖 [Reference](reference/)
**Information-oriented technical documentation**

Complete API and technical references:
- [Models](reference/models.md) - All available components and their parameters
- [Configuration](reference/config.md) - Configuration system reference
- [Data](reference/data.md) - Data handling utilities
- [API](reference/api.md) - Complete API documentation

### 💡 [Explanation](explanation/)
**Understanding-oriented discussions**

Background and conceptual explanations:
- [Architecture](explanation/architecture.md) - System design principles
- [SSL Methods](explanation/ssl-methods.md) - Self-supervised learning approaches
- [Performance](explanation/performance.md) - Optimization strategies

## Building Documentation Locally

### Prerequisites
- Python 3.8+
- MkDocs and dependencies

### Setup
```bash
# Install documentation requirements
pip install -r docs/requirements.txt

# Build documentation
cd docs
mkdocs build

# Serve locally with live reload
mkdocs serve
```

The documentation will be available at `http://localhost:8000`

## Documentation Standards

### Content Guidelines

**Tutorials** (Learning-oriented):
- Start with "What you'll learn"
- Include time estimates
- Use safe-to-fail examples
- Provide checkpoints and validation steps
- End with "What's next"

**How-to Guides** (Problem-oriented):
- Structure as "Problem: Goal → Solution"
- Focus on practical, actionable steps
- Include multiple solution approaches
- Use real-world scenarios

**Reference** (Information-oriented):
- Complete parameter documentation
- Working code examples
- Accurate technical specifications
- Consistent formatting

**Explanation** (Understanding-oriented):
- Focus on "why" not "how"
- Provide context and background
- Discuss design decisions
- Connect concepts together

### Style Guidelines

- Use clear, concise language
- Include working code examples
- Add time estimates for tutorials
- Use consistent formatting
- Provide cross-references between sections

## Contributing to Documentation

### Quick Contribution
1. Fork the repository
2. Edit documentation files in `docs/`
3. Test locally with `mkdocs serve`
4. Submit pull request

### Major Changes
1. Follow the Diátaxis framework principles
2. Update navigation in `docs/index.md` if needed
3. Ensure all links work
4. Test with different scenarios

### File Organization
- Keep examples working and up-to-date
- Use relative links between docs
- Place images in `docs/assets/`
- Follow existing naming conventions

## Getting Help

- 📖 **Documentation Issues**: [GitHub Issues](https://github.com/yourusername/tabular-ssl/issues)
- 💬 **Content Questions**: [GitHub Discussions](https://github.com/yourusername/tabular-ssl/discussions)
- ✨ **Improvements**: Submit pull requests with specific changes

## License

Documentation is licensed under the same license as the Tabular SSL library. 