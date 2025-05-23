# Tabular SSL Documentation

This directory contains the documentation for the Tabular SSL library, built using MkDocs Material and following the Diátaxis framework.

## Structure

The documentation is organized into four main sections:

1. **Tutorials**: Step-by-step guides for getting started
2. **How-to Guides**: Practical solutions for specific tasks
3. **Reference**: Technical API documentation
4. **Explanation**: Background information and concepts

## Building the Documentation

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. Install the required packages:

```bash
pip install -r requirements.txt
```

### Building Locally

To build and serve the documentation locally:

```bash
mkdocs serve
```

This will start a local server at `http://127.0.0.1:8000` where you can preview the documentation.

### Building for Production

To build the documentation for production:

```bash
mkdocs build
```

This will create a `site` directory with the built documentation.

## Contributing to Documentation

1. Make your changes in the appropriate section
2. Test locally using `mkdocs serve`
3. Submit a pull request

## Documentation Guidelines

- Follow the Diátaxis framework structure
- Use clear and concise language
- Include code examples where appropriate
- Keep code examples up to date
- Use proper Markdown formatting
- Include cross-references to related sections

## License

The documentation is licensed under the same license as the main project. 