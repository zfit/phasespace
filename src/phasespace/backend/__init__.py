"""Backend abstraction for phasespace.

This module provides backend-agnostic function decorators and utilities for different computational backends
(TensorFlow, NumPy).
"""

from __future__ import annotations

import os
import warnings
from enum import Enum, auto
from typing import Literal

__all__ = [
    "BACKEND",
    "Tensor",
    "Variable",
    "assert_equal",
    "assert_greater_equal",
    "function",
    "function_jit",
    "function_jit_fixedshape",
    "get_backend",
    "get_shape",
    "random",
    "set_backend",
    "tnp",
]


class BackendType(Enum):
    TENSORFLOW = auto()
    NUMPY = auto()

    @staticmethod
    def from_string(backend: str) -> BackendType:
        """Convert string to BackendType enum.

        Args:
            backend: Backend name string.

        Returns:
            Corresponding BackendType enum value.

        Raises:
            NotImplementedError: If backend is not recognized.
        """
        backend_formatted = backend.lower().strip()
        if backend_formatted in {"", "np", "numpy"}:
            return BackendType.NUMPY
        if backend_formatted in {"tf", "tensorflow"}:
            return BackendType.TENSORFLOW
        raise NotImplementedError(f'No backend implemented for "{backend}"')


# Module-level variables that will be set by _initialize_backend
BACKEND: BackendType = None  # type: ignore[assignment]
tnp = None
random = None
function = None
function_jit = None
function_jit_fixedshape = None
Tensor = None
Variable = None
get_shape = None
assert_equal = None
assert_greater_equal = None

_initialized = False


def _initialize_tensorflow() -> None:
    """Initialize TensorFlow backend."""
    global tnp, random, function, function_jit, function_jit_fixedshape
    global Tensor, Variable, get_shape, assert_equal, assert_greater_equal

    import tensorflow as tf
    import tensorflow.experimental.numpy as _tnp

    from . import _tf_random

    tnp = _tnp
    random = _tf_random

    RELAX_SHAPES = True

    function = tf.function(autograph=False, jit_compile=False)
    function_jit = tf.function(
        autograph=False, reduce_retracing=RELAX_SHAPES, jit_compile=True
    )
    function_jit_fixedshape = tf.function(
        autograph=False, reduce_retracing=False, jit_compile=True
    )

    Tensor = tf.Tensor
    Variable = tf.Variable
    get_shape = tf.shape
    assert_equal = tf.assert_equal
    assert_greater_equal = tf.debugging.assert_greater_equal

    is_eager = bool(os.environ.get("PHASESPACE_EAGER"))
    tf.config.run_functions_eagerly(is_eager)


def _initialize_numpy() -> None:
    """Initialize NumPy backend."""
    global tnp, random, function, function_jit, function_jit_fixedshape
    global Tensor, Variable, get_shape, assert_equal, assert_greater_equal

    import numpy as _np

    from . import _np_random

    tnp = _np
    random = _np_random

    function = lambda x: x  # noqa: E731
    function_jit = lambda x: x  # noqa: E731
    function_jit_fixedshape = lambda x: x  # noqa: E731

    Tensor = _np.ndarray
    Variable = _np.ndarray
    get_shape = _np.shape

    def _assert_equal(x, y, message: str, name: str = "") -> None:
        return _np.testing.assert_equal(x, y, err_msg=message)

    def _assert_greater_equal(x, y, message: str, name: str = "") -> None:
        return _np.testing.assert_array_less(-x, -y, err_msg=message)

    assert_equal = _assert_equal
    assert_greater_equal = _assert_greater_equal


def _initialize_backend(backend_type: BackendType) -> None:
    """Initialize the specified backend.

    Args:
        backend_type: The backend to initialize.
    """
    global BACKEND, _initialized

    if backend_type == BackendType.TENSORFLOW:
        _initialize_tensorflow()
    elif backend_type == BackendType.NUMPY:
        _initialize_numpy()
    else:
        raise NotImplementedError(f"Backend {backend_type} not implemented")

    BACKEND = backend_type
    _initialized = True


def get_backend() -> BackendType:
    """Get the current backend.

    Returns:
        The current BackendType enum value.

    Example:
        >>> from phasespace.backend import get_backend
        >>> get_backend()
        <BackendType.NUMPY: 2>
    """
    return BACKEND


def set_backend(
    backend: Literal["tensorflow", "tf", "numpy", "np"] | BackendType,
) -> None:
    """Set the computational backend.

    This function allows switching backends at runtime. Note that switching
    backends after generating events may cause issues with cached graphs
    in TensorFlow.

    Args:
        backend: Backend to use. Can be a string ("tensorflow", "tf", "numpy",
            "np") or a BackendType enum value.

    Raises:
        NotImplementedError: If the backend is not recognized.

    Example:
        >>> from phasespace.backend import set_backend, get_backend
        >>> set_backend("tensorflow")
        >>> get_backend()
        <BackendType.TENSORFLOW: 1>
    """
    global _initialized

    if isinstance(backend, str):
        backend_type = BackendType.from_string(backend)
    else:
        backend_type = backend

    if _initialized and BACKEND == backend_type:
        return  # Already using this backend

    if _initialized:
        warnings.warn(
            f"Switching backend from {BACKEND.name} to {backend_type.name}. "
            "This may cause issues with cached TensorFlow graphs. "
            "It's recommended to set the backend before importing phasespace modules.",
            UserWarning,
            stacklevel=2,
        )

    _initialize_backend(backend_type)


# Initialize backend on module import based on environment variable
_initial_backend = BackendType.from_string(os.environ.get("PHASESPACE_BACKEND", ""))
_initialize_backend(_initial_backend)
