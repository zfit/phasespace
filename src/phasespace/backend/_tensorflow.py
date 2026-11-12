"""TensorFlow implementation of the backend interface."""

import os

import tensorflow as tf
import tensorflow.experimental.numpy as tnp

from ._types import Implementation, Random


def create_implementation() -> Implementation:
    tf.config.run_functions_eagerly(
        os.environ.get("PHASESPACE_EAGER", "0") not in {"", "0"}
    )
    return Implementation(
        array=tnp,
        random=Random(
            Generator=tf.random.Generator,
            default_rng=tf.random.get_global_generator,
            from_seed=tf.random.Generator.from_seed,
            uniform=uniform,
        ),
        Tensor=tf.Tensor,
        Variable=tf.Variable,
        shape=tf.shape,
        assert_equal=tf.debugging.assert_equal,
        assert_greater_equal=tf.debugging.assert_greater_equal,
    )


def uniform(rng, shape, minval, maxval, dtype):
    return rng.uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype)
