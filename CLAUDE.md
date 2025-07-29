# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
```bash
python -m pytest                                                    # Run all tests
python -m pytest tests/test_model.py                               # Run specific test file
python -m pytest --cov=lsbi --cov-report=term-missing             # Run tests with coverage
```

### Code Quality
```bash
black .                                                            # Format code
isort --profile black .                                            # Sort imports
pydocstyle --convention=numpy lsbi                                 # Check docstring style
```

### Installation & Setup
```bash
python -m pip install .                                           # Install package
python -m pip install -e .                                        # Install in editable mode
python -m pip install ".[test]"                                   # Install with test dependencies
python -m pip install ".[all,docs]"                               # Install with all dependencies
```

### Documentation
```bash
cd docs && make html                                               # Build documentation
sphinx-apidoc -fM -t docs/templates/ -o docs/source/ lsbi/       # Regenerate API docs
```

## Architecture

### Core Package Structure
- `lsbi/model.py` - Main linear models (`LinearModel`, `MixtureModel`, `ReducedLinearModel`)
- `lsbi/stats.py` - Statistics utilities (`multivariate_normal`, `mixture_normal`, `dkl`)
- `lsbi/utils.py` - Utility functions (`logdet`, `bisect`, `dediagonalise`, `alias`)

### Key Model Classes

**LinearModel**: The core multilinear Bayesian model implementing:
- D|θ ~ N(m + Mθ, C) - Data likelihood  
- θ ~ N(μ, Σ) - Parameter prior
- Provides methods for `likelihood()`, `prior()`, `posterior()`, `evidence()`, `joint()`
- Supports diagonal covariance optimizations via `diagonal_M`, `diagonal_C`, `diagonal_Σ` flags

**MixtureModel**: Extends LinearModel for categorical mixture components:
- Adds mixture weights via `logw` parameter
- All distributions become `mixture_normal` instances
- Requires Monte Carlo estimation for KL divergence (`dkl()` with `n>0`)

**ReducedLinearModel**: Parameter-space only model for Gaussian likelihoods:
- Defined by likelihood peak `mu_L`, covariance `Sigma_L`, and maximum `logLmax`
- Direct posterior computation without explicit data modeling

### Statistics Framework

**multivariate_normal**: Vectorized multivariate normal distributions supporting:
- Broadcasting across distribution parameters (`mean`, `cov`)
- Diagonal covariance optimization
- Methods: `logpdf()`, `pdf()`, `rvs()`, `predict()`, `marginalise()`, `condition()`
- Bijector transformations between hypercube and physical space

**mixture_normal**: Extends multivariate_normal for mixture models:
- Additional `logw` mixture weights parameter
- Joint probability computation via `joint=True` parameter

### Development Notes

- All models support broadcasting for vectorized operations
- Diagonal optimizations are automatically detected based on array dimensions
- Greek letter parameters support both Unicode (μ, Σ) and ASCII (mu, Sigma) aliases via `alias()` utility
- Extensive test coverage across different dimensions, shapes, and diagonal configurations
- Package uses modern Python packaging with `pyproject.toml` and setuptools build backend