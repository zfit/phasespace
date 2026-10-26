__all__ = [
    "Generator",
    "default_rng",
    "from_seed",
    "uniform",
]

import secrets
from numbers import Integral

import jax
import jax.numpy as jnp


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


from_seed = Generator


def default_rng(seed=None):
    return seed if isinstance(seed, Generator) else Generator(seed)


def uniform(rng, shape, minval, maxval, dtype):
    return rng.uniform(low=minval, high=maxval, size=shape).astype(dtype)
