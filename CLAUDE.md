# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PhaseSpace is a TensorFlow-based Python package for n-body phase space generation in High Energy Physics (HEP). It implements the Raubold and Lynch method (GENBOD function from CERNLIB) for generating particle decay events. The package uses TensorFlow 2.x in eager mode as its computational backend.

## Environment Setup

This project uses conda environments. Before working:

```bash
source /path/to/conda/activate
conda activate phasespace  # or appropriate environment name
```

Install the package in development mode:

```bash
pip install -e .[dev]
```

For the `fromdecay` functionality (integration with DecayLanguage):

```bash
pip install -e .[fromdecay]
```

## Testing

Run tests using pytest with parallel execution:

```bash
pytest -s -n auto
```

Run a specific test file:

```bash
pytest -k test_generate
```

Run with coverage:

```bash
coverage run -m pytest --basetemp={envtmpdir}
coverage xml
```

The `PHASESPACE_EAGER` environment variable controls TensorFlow's eager execution mode during tests:

```bash
PHASESPACE_EAGER=1 pytest -n auto tests
```

Physics validation tests produce plots in `tests/plots/` directory.

### Test Data

Before running tests, download test data:

```bash
python data/download_test_files.py
```

## Code Quality

Run pre-commit hooks:

```bash
pre-commit run -a
```

The project uses:
- `ruff` for linting and formatting (max line length: 120)
- `isort` for import sorting
- `pyupgrade` for Python 3.10+ syntax
- `nbstripout` for Jupyter notebook cleanup

## Architecture

### Core Components

- **`src/phasespace/phasespace.py`**: Core implementation
  - `GenParticle`: Main class representing particles in decay chains
  - `nbody_decay()`: Shortcut function for simple n-body decays
  - `generate()`: Returns TensorFlow tensors (essentially numpy arrays in eager mode)
  - `generate_tensor()`: Returns TensorFlow graph operations

- **`src/phasespace/kinematics.py`**: Kinematic calculations and transformations

- **`src/phasespace/backend.py`**: TensorFlow function decorators
  - `function`: Standard TF function wrapper
  - `function_jit`: JIT-compiled functions with shape relaxation
  - `function_jit_fixedshape`: JIT-compiled functions without shape relaxation

- **`src/phasespace/random.py`**: Random number generation utilities

- **`src/phasespace/fromdecay/`**: Integration with DecayLanguage package
  - `GenMultiDecay`: High-level interface for particles with multiple decay modes
  - `mass_functions.py`: Mass distribution functions for resonances
  - Requires: `particle`, `zfit`, `zfit-physics`, `decaylanguage`

### Key Design Patterns

1. **Particle Representation**: `GenParticle` instances can have:
   - Fixed masses (float/array) or variable masses (callable functions)
   - Children particles (decay products) set via `set_children()`
   - Hierarchical decay chains built by nesting particles

2. **Mass Functions**: Particles can have mass distributions instead of fixed masses. The mass function must return a TensorFlow tensor with shape `(nevents,)`.

3. **Graph Caching**: `GenParticle` caches TensorFlow graphs for efficient repeated generation in loops.

4. **Eager vs Graph Mode**: Controlled by `PHASESPACE_EAGER` environment variable. The package primarily uses eager mode (TF 2.x default).

## Documentation

Build documentation:

```bash
cd docs
make html
```

Documentation uses:
- Sphinx with bootstrap theme
- MyST-NB for Jupyter notebook integration
- Notebooks in `docs/` are validated during CI with `nbval`

## Common Development Tasks

### Running Benchmarks

```bash
python benchmark/bench_phasespace.py
```

### Validating Against Reference Implementations

Physics validation compares results against:
- TGenPhaseSpace (ROOT) for simple n-body decays
- RapidSim for sequential decays

Test files in `tests/helpers/` contain reference implementations.

## Important Conventions

1. **Default Values**: Always use `None` for function defaults, then assign actual defaults with `if arg is None:`. Document the default value in docstrings.

2. **Container Testing**: Test for `Collection` or `Container` instead of `(list, tuple)`.

3. **Error Handling**: Fail loudly. Raise errors for invalid inputs or missing functionality. Don't catch exceptions and make assumptions.

4. **Backwards Compatibility**: Breaking changes are acceptable unless explicitly told to preserve compatibility.

5. **Correctness First**: Never compromise correctness for performance or convenience. If modifications require trade-offs with correctness, always ask first.

## Dependencies

Core requirements:
- Python ≥ 3.10
- TensorFlow ≥ 2.19.0
- TensorFlow Probability ≥ 0.25.0

Optional dependencies:
- `fromdecay`: particle, zfit, zfit-physics, decaylanguage
- `vector`: vector ≥ 1.0.0

## CI/CD

GitHub Actions runs:
- Tests on Ubuntu with Python 3.10, 3.11, 3.12, and 3.13
- Both eager modes (0 and 1)
- Parallel test execution with `pytest-xdist`
- Coverage reporting to Codecov
- Documentation notebook validation

Deployment to PyPI is automatic on tagged releases.
