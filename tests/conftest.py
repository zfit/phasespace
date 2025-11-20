"""Shared pytest fixtures for phasespace tests."""

import importlib
import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "helpers"))


def _check_backend_available(backend_name):
    """Check if a backend is available."""
    if backend_name == "tensorflow":
        try:
            import tensorflow  # noqa: F401

            return True
        except ImportError:
            return False
    elif backend_name == "numpy":
        return True  # Always available
    return False


# Create list of available backends for parametrization
AVAILABLE_BACKENDS = [b for b in ["numpy", "tensorflow"] if _check_backend_available(b)]


@pytest.fixture
def backend_context():
    """Context manager for switching backends in tests."""
    original_backend = os.environ.get("PHASESPACE_BACKEND", "")

    def _set_backend(backend_name):
        os.environ["PHASESPACE_BACKEND"] = backend_name
        # Reload the backend module to pick up the new setting
        import phasespace.backend

        importlib.reload(phasespace.backend)
        # Also reload phasespace modules that use the backend
        import phasespace.kinematics
        import phasespace.random

        import phasespace.phasespace

        importlib.reload(phasespace.random)
        importlib.reload(phasespace.kinematics)
        importlib.reload(phasespace.phasespace)
        import phasespace

        importlib.reload(phasespace)

    yield _set_backend

    # Restore original backend
    os.environ["PHASESPACE_BACKEND"] = original_backend


def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "tf_only: mark test as TensorFlow-only (skip for other backends)"
    )
