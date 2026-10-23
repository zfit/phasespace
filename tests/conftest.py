"""Shared pytest fixtures for phasespace tests."""

import os
import sys

import pytest

import phasespace

sys.path.append(os.path.join(os.path.dirname(__file__), "helpers"))


@pytest.fixture
def backend_context():
    """Switch through the public API and restore the original backend after each test."""
    original_backend = phasespace.get_backend()
    yield phasespace.set_backend
    phasespace.set_backend(original_backend)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "tf_only: mark test as TensorFlow-only (skip for other backends)"
    )
