__all__ = [
    "Generator",
    "default_rng",
    "from_seed",
    "uniform",
]

from numpy.random import PCG64, BitGenerator, Generator, default_rng


def from_seed(seed, alg: type[BitGenerator] | None = None) -> Generator:
    """Function that mimics `tf.random.Generator.from_seed`."""
    if alg is None:
        alg = PCG64
    bit_generator = alg(seed)
    return Generator(bit_generator)


def uniform(rng, shape, minval, maxval, dtype):
    return rng.uniform(low=minval, high=maxval, size=shape).astype(dtype)
