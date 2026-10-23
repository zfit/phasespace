"""Runtime backend selection and compilation for NumPy, TensorFlow, and JAX."""

from __future__ import annotations

import functools
import inspect
import os
from enum import Enum, auto
from types import SimpleNamespace
from typing import Literal

import numpy as np

__all__ = [
    "BACKEND",
    "BackendType",
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
    JAX = auto()

    @staticmethod
    def from_string(backend: str) -> BackendType:
        backend = backend.lower().strip()
        if backend in {"", "tf", "tensorflow"}:
            return BackendType.TENSORFLOW
        if backend in {"np", "numpy"}:
            return BackendType.NUMPY
        if backend == "jax":
            return BackendType.JAX
        raise NotImplementedError(f'No backend implemented for "{backend}"')


class _BackendModule:
    """Resolve attributes at use time so imported namespaces survive backend switches."""

    def __init__(self, name):
        self._name = name

    def __getattr__(self, name):
        return getattr(getattr(_implementation, self._name), name)


tnp = _BackendModule("array")
random = _BackendModule("random")
BACKEND = None
_generation = 0


def _initialize_backend(backend_type):
    if backend_type == BackendType.TENSORFLOW:
        import tensorflow as tf
        import tensorflow.experimental.numpy as tfnp

        from . import _tf_random

        tf.config.run_functions_eagerly(
            os.environ.get("PHASESPACE_EAGER", "0") not in {"", "0"}
        )
        return SimpleNamespace(
            array=tfnp,
            random=_tf_random,
            Tensor=tf.Tensor,
            Variable=tf.Variable,
            shape=tf.shape,
            assert_equal=tf.assert_equal,
            assert_greater_equal=tf.debugging.assert_greater_equal,
        )
    if backend_type == BackendType.NUMPY:
        from . import _np_random

        return SimpleNamespace(
            array=np, random=_np_random, Tensor=np.ndarray, Variable=np.ndarray
        )
    if backend_type == BackendType.JAX:
        import jax
        import jax.numpy as jnp

        from . import _jax_random

        jax.config.update("jax_enable_x64", True)
        return SimpleNamespace(
            array=jnp, random=_jax_random, Tensor=jax.Array, Variable=jax.Array
        )
    raise NotImplementedError(f"Backend {backend_type} not implemented")


def _decorate(func=None, *, jit=False, fixed_shape=False, static_argnames=()):
    if func is None:
        return functools.partial(
            _decorate, jit=jit, fixed_shape=fixed_shape, static_argnames=static_argnames
        )
    compiled = None
    generation = None
    is_method = next(iter(inspect.signature(func).parameters), None) == "self"

    @functools.wraps(func)
    def dispatch(*args, **kwargs):
        nonlocal compiled, generation
        if generation != _generation:
            if BACKEND == BackendType.TENSORFLOW:
                import tensorflow as tf

                compiled = tf.function(
                    func,
                    autograph=False,
                    jit_compile=jit,
                    reduce_retracing=jit and not fixed_shape,
                )
            elif BACKEND == BackendType.JAX and jit:
                import jax

                compiled = jax.jit(func, static_argnames=static_argnames)
            else:
                compiled = func
            generation = _generation
        if BACKEND == BackendType.TENSORFLOW and is_method:
            return compiled.__get__(args[0], type(args[0]))(*args[1:], **kwargs)
        return compiled(*args, **kwargs)

    return dispatch


function = _decorate
function_jit = functools.partial(_decorate, jit=True)
function_jit_fixedshape = functools.partial(_decorate, jit=True, fixed_shape=True)


def get_shape(array):
    if BACKEND == BackendType.TENSORFLOW:
        return _implementation.shape(array)
    return np.shape(array)


def assert_equal(x, y, message="", name=None):
    if BACKEND == BackendType.TENSORFLOW:
        return _implementation.assert_equal(x, y, message=message, name=name)
    np.testing.assert_equal(np.asarray(x), np.asarray(y), err_msg=message)


def assert_greater_equal(x, y, message="", name=None):
    if BACKEND == BackendType.TENSORFLOW:
        return _implementation.assert_greater_equal(x, y, message=message, name=name)
    np.testing.assert_equal(np.asarray(x) >= np.asarray(y), True, err_msg=message)


def get_backend() -> BackendType:
    """Return the currently selected backend."""
    return BACKEND


def set_backend(
    backend: Literal["tensorflow", "tf", "numpy", "np", "jax"] | BackendType,
) -> None:
    """Select the backend for subsequent calls, including existing decay objects.

    TensorFlow is the default. JAX requires the optional ``jax`` extra and uses double precision. Backend
    selection is process-wide; switch outside compiled user functions and do not switch while another thread
    is generating events. Existing arrays and random generators retain their original backend.
    """
    global BACKEND, Tensor, Variable, _implementation, _generation
    backend_type = (
        BackendType.from_string(backend) if isinstance(backend, str) else backend
    )
    if not isinstance(backend_type, BackendType):
        raise NotImplementedError(f"Backend {backend_type} not implemented")
    if BACKEND == backend_type:
        return
    implementation = _initialize_backend(backend_type)
    _implementation = implementation
    Tensor = implementation.Tensor
    Variable = implementation.Variable
    BACKEND = backend_type
    _generation += 1


set_backend(os.environ.get("PHASESPACE_BACKEND", "tensorflow"))
