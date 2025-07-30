# Documentation

This directory contains additional documentation for the Explainable AI Quality Inspection project.

## Contents

- **API Documentation**: Detailed documentation of all classes and functions
- **User Guide**: Step-by-step guides for different use cases
- **Technical Notes**: Architecture decisions and implementation details
- **Examples**: Code examples and tutorials

## Generating Documentation

To generate API documentation:

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Generate documentation
cd docs
sphinx-quickstart
make html
```

## Contributing to Documentation

When adding new features or modifying existing code:

1. Update docstrings following Google style
2. Add examples where appropriate
3. Update this README if adding new documentation types
4. Keep documentation in sync with code changes