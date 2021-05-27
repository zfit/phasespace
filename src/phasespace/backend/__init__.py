"""Backend abstraction for phasespace.

This module provides backend-agnostic function decorators and utilities for different
computational backends (TensorFlow, NumPy).
"""

import os
from enum import Enum, auto

__all__ = [
    "Tensor",
    "Variable",
    "assert_equal",
    "assert_greater_equal",
    "function",
    "function_jit",
    "function_jit_fixedshape",
    "get_shape",
    "random",
    "tnp",
]


class BackendType(Enum):
    TENSORFLOW = auto()
    NUMPY = auto()

    @staticmethod
    def get_backend(backend: str) -> "BackendType":
        backend_formatted = backend.lower().strip()
        if backend_formatted in {"", "np", "numpy"}:
            return BackendType.NUMPY
        if backend_formatted in {"tf", "tensorflow"}:
            return BackendType.TENSORFLOW
        raise NotImplementedError(f'No backend implemented for "{backend}"')


BACKEND = BackendType.get_backend(os.environ.get("PHASESPACE_BACKEND", ""))

if BACKEND == BackendType.TENSORFLOW:
    try:
        import tensorflow as tf
    except ImportError:
        message = """
To use phasespace with tensorflow as backend, please reinstall with:

    pip install phasespace[tf]
"""
        raise ImportError(message)
    import tensorflow.experimental.numpy as tnp

    from . import _tf_random as random

    #: Whether to enable shape relaxation for JIT-compiled functions
    RELAX_SHAPES = True

    #: Standard TensorFlow function wrapper without JIT compilation
    function = tf.function(autograph=False, jit_compile=False)

    #: JIT-compiled TensorFlow function with shape relaxation enabled
    function_jit = tf.function(
        autograph=False, reduce_retracing=RELAX_SHAPES, jit_compile=True
    )

    #: JIT-compiled TensorFlow function without shape relaxation
    function_jit_fixedshape = tf.function(
        autograph=False, reduce_retracing=False, jit_compile=True
    )

    # Type aliases for backend abstraction
    Tensor = tf.Tensor
    Variable = tf.Variable

    # Get shape dynamically (for graph mode compatibility)
    get_shape = tf.shape
    assert_equal = tf.assert_equal
    assert_greater_equal = tf.debugging.assert_greater_equal

    # Set eager mode based on environment variable
    is_eager = bool(os.environ.get("PHASESPACE_EAGER"))
    tf.config.run_functions_eagerly(is_eager)

if BACKEND == BackendType.NUMPY:
    import numpy as tnp

    from . import _np_random as random

    function = lambda x: x  # noqa: E731
    function_jit = lambda x: x  # noqa: E731
    function_jit_fixedshape = lambda x: x  # noqa: E731

    Tensor = tnp.ndarray
    Variable = tnp.ndarray
    get_shape = tnp.shape

    def assert_equal(x, y, message: str, name: str = "") -> None:
        return tnp.testing.assert_equal(x, y, err_msg=message)

    def assert_greater_equal(x, y, message: str, name: str = "") -> None:
        return tnp.testing.assert_array_less(-x, -y, err_msg=message)
