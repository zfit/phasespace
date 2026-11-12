"""Runtime backend selection and compilation for NumPy, TensorFlow, and JAX.

Other `phasespace` sub-modules import from this module and `set_backend` decides which library executes it.
Because the backend can be switched after other modules imported names from here, every public name looks up
the active backend when it is used rather than when it is imported.
"""

from __future__ import annotations

__all__ = [
    "BACKEND",
    "BackendType",
    "Tensor",
    "Variable",
    "as_event_count",
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

import functools
import inspect
import os
from enum import Enum, auto
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal

import numpy as np


class BackendType(Enum):
    NUMPY = auto()
    JAX = auto()
    TENSORFLOW = auto()

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
        return getattr(getattr(_active.implementation, self._name), name)


_active = SimpleNamespace(
    backend=None,
    implementation=None,
)
"""Single source of the active backend type and its implementation namespace, set by `set_backend`.

Nothing else in the package stores a copy of this state, so that a backend switch reaches every caller.
"""

tnp = _BackendModule("array")
"""NumPy-like array API of the active backend."""

random = _BackendModule("random")
"""Random number API of the active backend."""

if TYPE_CHECKING:
    BACKEND: BackendType
    Tensor: type
    Variable: type


def __getattr__(name):
    """Serve `BACKEND`, `Tensor` and `Variable` from the active backend at lookup time.

    Module attribute access such as ``backend.Tensor`` therefore follows `set_backend`. A direct import such
    as ``from phasespace.backend import Tensor`` still binds the value at import time.
    """
    if name == "BACKEND":
        return _active.backend
    if name in {"Tensor", "Variable"}:
        return getattr(_active.implementation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
            assert_equal=tf.debugging.assert_equal,
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


def _compile(func, backend_type, jit, fixed_shape, static_argnames):
    if backend_type == BackendType.TENSORFLOW:
        import tensorflow as tf

        return tf.function(
            func,
            autograph=False,
            jit_compile=jit,
            reduce_retracing=jit and not fixed_shape,
        )
    if backend_type == BackendType.JAX and jit:
        import jax

        return jax.jit(func, static_argnames=static_argnames)
    return func


def _decorate(func=None, *, jit=False, fixed_shape=False, static_argnames=()):
    """Compile ``func`` for the active backend on first call, once per backend.

    Each compiled version is kept, so switching back to a backend that was used before does not recompile.
    """
    if func is None:
        return functools.partial(
            _decorate, jit=jit, fixed_shape=fixed_shape, static_argnames=static_argnames
        )
    compiled = {}
    is_method = next(iter(inspect.signature(func).parameters), None) == "self"

    @functools.wraps(func)
    def dispatch(*args, **kwargs):
        backend_type = _active.backend
        if backend_type not in compiled:
            compiled[backend_type] = _compile(
                func, backend_type, jit, fixed_shape, static_argnames
            )
        compiled_func = compiled[backend_type]
        if backend_type == BackendType.TENSORFLOW and is_method:
            return compiled_func.__get__(args[0], type(args[0]))(*args[1:], **kwargs)
        return compiled_func(*args, **kwargs)

    return dispatch


function = _decorate
function_jit = functools.partial(_decorate, jit=True)
function_jit_fixedshape = functools.partial(_decorate, jit=True, fixed_shape=True)


def as_event_count(n_events):
    if _active.backend != BackendType.TENSORFLOW:
        return int(n_events)
    if isinstance(n_events, _active.implementation.Variable):
        return n_events
    return tnp.asarray(n_events, dtype=tnp.int64)


def get_shape(array):
    if _active.backend == BackendType.TENSORFLOW:
        return _active.implementation.shape(array)
    return np.shape(array)


def assert_equal(x, y, message="", name=None):
    if _active.backend == BackendType.TENSORFLOW:
        return _active.implementation.assert_equal(x, y, message=message, name=name)
    np.testing.assert_equal(np.asarray(x), np.asarray(y), err_msg=message)


def assert_greater_equal(x, y, message="", name=None):
    if _active.backend == BackendType.TENSORFLOW:
        return _active.implementation.assert_greater_equal(
            x, y, message=message, name=name
        )
    np.testing.assert_equal(np.asarray(x) >= np.asarray(y), True, err_msg=message)


def get_backend() -> BackendType:
    """Return the currently selected backend."""
    return _active.backend


def set_backend(
    backend: Literal["tensorflow", "tf", "numpy", "np", "jax"] | BackendType,
) -> None:
    """Select the backend for subsequent calls, including existing decay objects.

    TensorFlow is the default. JAX requires the optional ``jax`` extra and uses double precision. Backend
    selection is process-wide; switch outside compiled user functions and do not switch while another thread
    is generating events. Existing arrays and random generators retain their original backend.
    """
    backend_type = (
        BackendType.from_string(backend) if isinstance(backend, str) else backend
    )
    if not isinstance(backend_type, BackendType):
        raise NotImplementedError(f"Backend {backend_type} not implemented")
    if _active.backend == backend_type:
        return
    _active.implementation = _initialize_backend(backend_type)
    _active.backend = backend_type


set_backend(os.environ.get("PHASESPACE_BACKEND", "tensorflow"))
