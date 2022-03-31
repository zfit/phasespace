"""Random number generation.

As the random number generation is not a trivial thing, this module handles it uniformly.

It mimics the TensorFlows API on random generators and relies (currently) in global states on the TF states.
Especially on the global random number generator which will be used to get new generators.
"""
from __future__ import annotations

from phasespace.backend import random, tnp

SeedLike = int | random.Generator | None


def get_rng(seed: SeedLike = None) -> random.Generator:
    """Get or create random number generator of type `tf.random.Generator`.

    This can be used to either retrieve random number generators deterministically from them
    - global random number generator from TensorFlow,
    - from a random number generator generated from the seed or
    - from the random number generator passed.

    Both when using either the global generator or a random number generator is passed, they advance
    by exactly one step as `split` is called on them.

    Args:
        seed: This can be
          - `None` to get the global random number generator
          - a numerical seed to create a random number generator
          - a `tf.random.Generator`.

    Returns:
        A list of `tf.random.Generator`
    """
    if seed is None:
        rng = random.default_rng()
    elif not isinstance(seed, random.Generator):  # it's a seed, not an rng
        rng = random.from_seed(seed)
    else:
        rng = seed
    return rng


def generate_uniform(
    rng: random.Generator, shape: tuple[int, ...], minval=0, maxval=1, dtype=tnp.float64
) -> tnp.ndarray:
    try:
        return rng.uniform(shape, minval=minval, maxval=maxval, dtype=dtype)
    except TypeError:
        return rng.uniform(low=minval, high=maxval, size=shape).astype(dtype)
