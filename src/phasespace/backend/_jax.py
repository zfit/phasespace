"""JAX implementation of the backend interface."""

import secrets
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ._numpy import assert_equal, assert_greater_equal
from ._types import Implementation, Random


def create_implementation() -> Implementation:
    jax.config.update("jax_enable_x64", True)
    return Implementation(
        array=jnp,
        random=Random(
            Generator=Generator,
            default_rng=default_rng,
            from_seed=Generator,
            uniform=uniform,
        ),
        Tensor=jax.Array,
        Variable=jax.Array,
        shape=np.shape,
        assert_equal=assert_equal,
        assert_greater_equal=assert_greater_equal,
    )


class Generator:
    """Stateful generator facade over JAX's explicit random keys.

    Key advancement stays outside JIT-compiled kernels to avoid capturing random state during tracing.
    """

    def __init__(self, seed=None):
        if seed is None:
            seed = secrets.randbits(32)
        self._key = jax.random.key(seed)

    def uniform(self, low=0, high=1, size=None):
        self._key, key = jax.random.split(self._key)
        shape = (
            ()
            if size is None
            else (size,)
            if isinstance(size, Integral)
            else tuple(size)
        )
        return jax.random.uniform(
            key, shape=shape, dtype=jnp.float64, minval=low, maxval=high
        )


def default_rng(seed=None):
    return seed if isinstance(seed, Generator) else Generator(seed)


def uniform(rng, shape, minval, maxval, dtype):
    return rng.uniform(low=minval, high=maxval, size=shape).astype(dtype)
