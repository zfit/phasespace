"""Random number generation

As the random number generation is not a trivial thing, this module handles it uniformly.

It mimicks the TensorFlows API on random generators and relies (currently) in global states on the TF global states.
Especially on the global random number generator which will be used to get new generators.
"""
from typing import Union, Optional

import tensorflow as tf

SeedLike = Optional[Union[int, tf.random.Generator]]


def get_rng(seed: SeedLike = None) -> tf.random.Generator:
    """Get or create a random number generators of type `tf.random.Generator`.

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
        rng = tf.random.get_global_generator()
    elif not isinstance(seed, tf.random.Generator):  # it's a seed, not an rng
        rng = tf.random.Generator.from_seed(seed=seed)
    else:
        rng = seed
    return rng
