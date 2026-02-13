# Contributing to Causal Genomics Toolkit

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/causal-genomics-toolkit.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Install in development mode: `pip install -e .[dev]`

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Write docstrings for all public functions and classes (Google style)
- Keep line length to 100 characters maximum

### Testing

- Write unit tests for all new features
- Ensure all tests pass before submitting PR: `pytest tests/`
- Aim for >80% code coverage
- Add integration tests for complex workflows

### Documentation

- Update README.md if adding new features
- Add docstrings with usage examples
- Update examples/ directory with notebooks demonstrating new functionality

### Commit Messages

Use clear, descriptive commit messages:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions/modifications
- `refactor:` for code refactoring

Example: `feat: add colocalization analysis with eCAVIAR method`

## Submitting Changes

1. Ensure all tests pass
2. Update documentation
3. Push to your fork
4. Submit a Pull Request with:
   - Clear description of changes
   - Reference to related issues
   - Example usage if applicable

## Code Review Process

- All submissions require review
- Reviewers will check:
  - Code quality and style
  - Test coverage
  - Documentation
  - Performance implications

## Areas for Contribution

We welcome contributions in:

1. **New Methods**: Implement additional causal inference methods
2. **Data Loaders**: Support for more data formats and sources
3. **Visualization**: New plotting functions
4. **Performance**: Optimization of computational bottlenecks
5. **Documentation**: Examples, tutorials, and improved docs
6. **Testing**: Additional test cases and edge case handling

## Questions?

Open an issue or reach out to the maintainers!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
