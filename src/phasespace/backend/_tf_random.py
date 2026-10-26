__all__ = [
    "Generator",
    "default_rng",
    "from_seed",
    "uniform",
]

from tensorflow.random import Generator
from tensorflow.random import get_global_generator as default_rng

from_seed = Generator.from_seed


def uniform(rng, shape, minval, maxval, dtype):
    return rng.uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype)
