"""Random number generation of

As the random number generation is not a trivial thing, this module handles it uniformly.

It mimicks the TensorFlows API on random generators and relies (currently) in global states on the TF global states.
Especially on the global random number generator which will be used to get new generators.
"""
from typing import Union, Optional, List

import tensorflow as tf

SeedLike = Optional[Union[int, tf.random.Generator]]


def get_rng(
        seed: SeedLike = None,
        count: Optional[int] = 1) -> List[tf.random.Generator]:
    """Get or create a list of random number generators of type `tf.random.Generator`.

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

          This generator is split into `counts` generators which will be returned.

        count: how many random generators to create.

    Returns:
        A list of `tf.random.Generator`
    """
    if not tf.executing_eagerly():
        raise RuntimeError("Cannot get a new rng in Graph mode.")
    if seed is None:
        seed = tf.random.get_global_generator()
    if not isinstance(seed, tf.random.Generator):  # it's a seed, not an rng
        seed = tf.random.Generator.from_seed(seed=seed)

    rng = seed.split(count=count)
    return rng
