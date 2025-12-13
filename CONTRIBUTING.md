# Contributing to TSQVT

Thank you for your interest in contributing to TSQVT! This document provides guidelines for contributions.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect differing viewpoints and experiences

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

**Bug Report Template:**

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Import '...'
2. Call function '...'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
 - OS: [e.g., Ubuntu 22.04]
 - Python version: [e.g., 3.10.8]
 - TSQVT version: [e.g., 1.0.0]

**Additional context**
Any other context about the problem.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues.

**Enhancement Template:**

```markdown
**Is your feature request related to a problem?**
Clear description of the problem.

**Describe the solution you'd like**
Clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features.

**Additional context**
Any other context or screenshots.
```

### Pull Requests

1. **Fork the repo** and create your branch from `main`
2. **Follow the style guide** (see below)
3. **Write tests** for new features
4. **Update documentation** as needed
5. **Ensure tests pass** (`pytest tests/`)
6. **Submit your PR** with a clear description

## Development Setup

### 1. Clone and Install

```bash
git clone https://github.com/kerym/TSQVT.git
cd TSQVT
pip install -e ".[dev,docs,tests]"
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes

Follow the coding standards below.

### 4. Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=tsqvt tests/

# Run specific test file
pytest tests/test_gauge.py

# Run with verbose output
pytest -v tests/
```

### 5. Format Code

```bash
# Format with black
black tsqvt/ tests/

# Sort imports
isort tsqvt/ tests/

# Check style
flake8 tsqvt/ tests/

# Type checking
mypy tsqvt/
```

### 6. Build Documentation

```bash
cd docs/
make html
```

### 7. Commit and Push

```bash
git add .
git commit -m "Add: clear description of changes"
git push origin feature/your-feature-name
```

### 8. Create Pull Request

Open a PR on GitHub with:
- Clear title and description
- Link to related issues
- Screenshots/outputs if applicable

## Coding Standards

### Python Style Guide

We follow **PEP 8** with these specifics:

```python
# Line length: 88 characters (Black default)
# Indentation: 4 spaces
# Quotes: Double quotes for strings
# Imports: Absolute imports, grouped and sorted

# Good example:
import numpy as np
from tsqvt.core import SpectralManifold

def compute_coupling(energy: float, coupling_constant: float = 1.0) -> float:
    """
    Compute running coupling at given energy scale.
    
    Parameters
    ----------
    energy : float
        Energy scale in GeV.
    coupling_constant : float, optional
        Initial coupling constant (default: 1.0).
    
    Returns
    -------
    float
        Running coupling value.
    
    Examples
    --------
    >>> compute_coupling(91.1876, 1.0)
    0.0073
    """
    return coupling_constant / (1 + energy)
```

### Docstring Format

We use **NumPy style docstrings**:

```python
def function_name(param1: type1, param2: type2) -> return_type:
    """
    Short description (one line).
    
    Longer description if needed. Can span multiple lines
    and include mathematical formulas in LaTeX.
    
    Parameters
    ----------
    param1 : type1
        Description of param1.
    param2 : type2
        Description of param2.
    
    Returns
    -------
    return_type
        Description of return value.
    
    Raises
    ------
    ValueError
        If param1 is negative.
    
    See Also
    --------
    related_function : Related functionality.
    
    Notes
    -----
    Additional notes, algorithms, or mathematical details.
    
    References
    ----------
    .. [1] Author, "Title", Journal, Year.
    
    Examples
    --------
    >>> function_name(1, 2)
    3
    """
    pass
```

### Testing Guidelines

All new features require tests:

```python
import pytest
import numpy as np
from tsqvt.gauge import compute_C4_coefficient

def test_C4_coefficient_positive():
    """Test that C4 coefficient is positive."""
    result = compute_C4_coefficient(mass=100, charge=1.0)
    assert result > 0

def test_C4_coefficient_scaling():
    """Test quadratic scaling with mass."""
    result1 = compute_C4_coefficient(mass=100, charge=1.0)
    result2 = compute_C4_coefficient(mass=200, charge=1.0)
    np.testing.assert_allclose(result2 / result1, 4.0, rtol=1e-5)

@pytest.mark.parametrize("mass,expected", [
    (0, 0),
    (100, 0.023),
    (1000, 2.3),
])
def test_C4_coefficient_values(mass, expected):
    """Test known values."""
    result = compute_C4_coefficient(mass=mass, charge=1.0)
    np.testing.assert_allclose(result, expected, rtol=1e-3)
```

### Type Hints

Use type hints for all functions:

```python
from typing import Optional, Tuple, List
import numpy as np
from numpy.typing import NDArray

def run_rg(
    energy_range: Tuple[float, float],
    initial_coupling: float,
    n_steps: int = 1000,
    method: str = "RK45"
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run RG equations."""
    pass
```

## Documentation Guidelines

### README Files

Each subpackage should have a README explaining:
- Purpose of the subpackage
- Key classes and functions
- Usage examples
- References

### Theory Documentation

Mathematical theory should include:
- Clear definitions and notation
- Derivations of key formulas
- References to papers
- Examples with numerical values

### API Documentation

Generated automatically from docstrings using Sphinx.

## Git Workflow

### Branch Naming

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation only
- `test/description` - Test additions
- `refactor/description` - Code refactoring

### Commit Messages

Follow the format:

```
Type: Short description (50 chars or less)

Longer explanation if needed. Explain what and why,
not how. Wrap at 72 characters.

Fixes #123
```

**Types:**
- `Add:` New feature
- `Fix:` Bug fix
- `Docs:` Documentation
- `Test:` Tests
- `Refactor:` Code refactoring
- `Style:` Formatting
- `Perf:` Performance improvement

### Examples

```
Add: Implement 3-loop RG running

Implements 3-loop renormalization group equations
for gauge couplings with full threshold corrections.

Fixes #42
```

```
Fix: Correct sign error in C4 coefficient

The U(1) coefficient had wrong sign due to trace
convention. Now matches Connes formalism.

Fixes #87
```

## Testing Requirements

### Minimum Coverage

- New features: 90% coverage
- Bug fixes: 100% coverage of fixed code
- Overall: Maintain 85%+ coverage

### Test Types

1. **Unit tests** - Test individual functions
2. **Integration tests** - Test module interactions
3. **Regression tests** - Prevent known bugs
4. **Numerical tests** - Verify calculations
5. **Property tests** - Use Hypothesis for edge cases

## Documentation Requirements

### For New Features

- Docstrings for all public functions/classes
- Usage examples in docstrings
- Tutorial notebook if applicable
- Update README if needed
- Update API documentation

### For Bug Fixes

- Document the bug in commit message
- Add regression test
- Update CHANGELOG

## Review Process

### What Reviewers Look For

1. **Correctness** - Does it work as intended?
2. **Tests** - Are there adequate tests?
3. **Style** - Follows coding standards?
4. **Documentation** - Is it documented?
5. **Performance** - Any performance concerns?
6. **Breaking changes** - Backward compatibility?

### Timeline

- Initial review: Within 1 week
- Follow-up: Within 3 days
- Final approval: When all checks pass

## Release Process

Maintainers handle releases:

1. Update version in `setup.py`
2. Update CHANGELOG.md
3. Create release tag
4. Build and upload to PyPI
5. Update documentation

## Questions?

- Open an issue for questions
- Join discussions on GitHub
- Email maintainer: kerym.sanjuan@uned.es

## Acknowledgments

Thank you for contributing to TSQVT! Your work helps advance our understanding of quantum gravity and fundamental physics.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
