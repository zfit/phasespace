"""NumPy implementation of the backend interface."""

import numpy as np
from numpy.random import PCG64, BitGenerator, Generator, default_rng

from ._types import Implementation, Random


def create_implementation() -> Implementation:
    return Implementation(
        array=np,
        random=Random(
            Generator=Generator,
            default_rng=default_rng,
            from_seed=from_seed,
            uniform=uniform,
        ),
        Tensor=np.ndarray,
        Variable=np.ndarray,
        shape=np.shape,
        assert_equal=assert_equal,
        assert_greater_equal=assert_greater_equal,
    )


def assert_equal(x, y, message="", name=None):
    np.testing.assert_equal(np.asarray(x), np.asarray(y), err_msg=message)


def assert_greater_equal(x, y, message="", name=None):
    np.testing.assert_equal(np.asarray(x) >= np.asarray(y), True, err_msg=message)


def from_seed(seed, alg: type[BitGenerator] | None = None) -> Generator:
    """Function that mimics `tf.random.Generator.from_seed`."""
    if alg is None:
        alg = PCG64
    bit_generator = alg(seed)
    return Generator(bit_generator)


def uniform(rng, shape, minval, maxval, dtype):
    return rng.uniform(low=minval, high=maxval, size=shape).astype(dtype)
