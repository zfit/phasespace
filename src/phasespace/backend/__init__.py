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
    "array_api",
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
]

import functools
import inspect
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar, overload

import numpy as np

_P = ParamSpec("_P")
_R = TypeVar("_R")


def set_backend(
    backend: Literal["tensorflow", "tf", "numpy", "np", "jax"] | BackendType,
) -> None:
    """Select the backend for subsequent calls, including existing decay objects.

    The initial backend is taken from the ``PHASESPACE_BACKEND`` environment variable and defaults to
    TensorFlow. JAX requires the optional ``jax`` extra and uses double precision. Backend selection is
    process-wide; switch outside compiled user functions and do not switch while another thread is generating
    events. Existing arrays and random generators retain their original backend.
    """
    if isinstance(backend, str):
        backend_type = BackendType.from_string(backend)
    if not isinstance(backend_type, BackendType):
        raise NotImplementedError(f"Backend {backend_type} not implemented")
    if _active.backend == backend_type:
        return
    _active.implementation = _initialize_backend(backend_type)
    _active.backend = backend_type


def get_backend() -> BackendType:
    """Return the currently selected backend."""
    return _active.backend


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


@overload
def function(
    func: Callable[_P, _R],
    *,
    jit: bool = False,
    fixed_shape: bool = False,
    static_argnames: tuple[str, ...] = (),
) -> Callable[_P, _R]: ...
@overload
def function(
    func: None = None,
    *,
    jit: bool = False,
    fixed_shape: bool = False,
    static_argnames: tuple[str, ...] = (),
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...
def function(
    func: Callable[_P, _R] | None = None,
    *,
    jit: bool = False,
    fixed_shape: bool = False,
    static_argnames: tuple[str, ...] = (),
) -> Callable[_P, _R] | Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Compile ``func`` for the active backend on first call, once per backend.

    Each compiled version is kept, so switching back to a backend that was used before does not recompile.
    """
    if func is None:
        return functools.partial(
            function, jit=jit, fixed_shape=fixed_shape, static_argnames=static_argnames
        )
    compiled: dict[BackendType, Callable[_P, _R]] = {}
    is_method = next(iter(inspect.signature(func).parameters), None) == "self"

    @functools.wraps(func)
    def dispatch(*args: _P.args, **kwargs: _P.kwargs) -> _R:
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


function_jit = functools.partial(function, jit=True)
function_jit_fixedshape = functools.partial(function, jit=True, fixed_shape=True)


def as_event_count(n_events):
    if _active.backend != BackendType.TENSORFLOW:
        return int(n_events)
    if isinstance(n_events, _active.implementation.Variable):
        return n_events
    return array_api.asarray(n_events, dtype=array_api.int64)


def get_shape(array):
    return _active.implementation.shape(array)


def assert_equal(x, y, message="", name=None):
    return _active.implementation.assert_equal(x, y, message=message, name=name)


def assert_greater_equal(x, y, message="", name=None):
    return _active.implementation.assert_greater_equal(x, y, message=message, name=name)


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


class _BackendModule:
    """Resolve attributes at use time so imported namespaces survive backend switches."""

    def __init__(self, name):
        self._name = name

    def __getattr__(self, name):
        return getattr(getattr(_active.implementation, self._name), name)


if TYPE_CHECKING:
    # Type checkers see the NumPy versions, the API that all backends follow.
    import numpy as array_api
    from numpy import ndarray as Tensor
    from numpy import ndarray as Variable

    from . import _np_random as random

    BACKEND: BackendType
else:
    array_api = _BackendModule("array")
    """NumPy-compatible array module of the active backend, conventionally imported as ``xp``."""
    random = _BackendModule("random")
    """Random number API of the active backend."""


def _initialize_backend(backend_type: BackendType) -> _Implementation:
    if backend_type == BackendType.TENSORFLOW:
        import tensorflow as tf
        import tensorflow.experimental.numpy as tnp

        from . import _tf_random

        tf.config.run_functions_eagerly(
            os.environ.get("PHASESPACE_EAGER", "0") not in {"", "0"}
        )
        return _Implementation(
            array=tnp,
            random=_tf_random,
            Tensor=tf.Tensor,
            Variable=tf.Variable,
            shape=tf.shape,
            assert_equal=tf.debugging.assert_equal,
            assert_greater_equal=tf.debugging.assert_greater_equal,
        )
    if backend_type == BackendType.NUMPY:
        from . import _np_assert, _np_random

        return _Implementation(
            array=np,
            random=_np_random,
            Tensor=np.ndarray,
            Variable=np.ndarray,
            shape=np.shape,
            assert_equal=_np_assert.assert_equal,
            assert_greater_equal=_np_assert.assert_greater_equal,
        )
    if backend_type == BackendType.JAX:
        import jax
        import jax.numpy as jnp

        from . import _jax_random, _np_assert

        jax.config.update("jax_enable_x64", True)
        return _Implementation(
            array=jnp,
            random=_jax_random,
            Tensor=jax.Array,
            Variable=jax.Array,
            shape=np.shape,
            assert_equal=_np_assert.assert_equal,
            assert_greater_equal=_np_assert.assert_greater_equal,
        )
    raise NotImplementedError(f"Backend {backend_type} not implemented")


@dataclass(frozen=True)
class _Implementation:
    """Library-specific objects behind the public names of this module, one instance per backend."""

    array: ModuleType
    random: ModuleType
    Tensor: type
    Variable: type
    shape: Callable[[Any], Any]
    assert_equal: Callable[..., Any]
    assert_greater_equal: Callable[..., Any]


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


@dataclass
class _ActiveBackend:
    backend: BackendType
    implementation: _Implementation


_initial_backend = BackendType.from_string(
    os.environ.get("PHASESPACE_BACKEND", "tensorflow")
)
_active = _ActiveBackend(_initial_backend, _initialize_backend(_initial_backend))
"""Single source of the active backend type and its implementation, changed by `set_backend`.

Nothing else in the package stores a copy of this state, so that a backend switch reaches every caller.
"""
