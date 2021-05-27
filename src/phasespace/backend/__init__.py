"""TensorFlow backend function decorators for phasespace.

This module provides standardized TensorFlow function decorators for different compilation and optimization
strategies used throughout the phasespace package.
"""

import os

import tensorflow as tf
import tensorflow.experimental.numpy as tnp

from . import _tf_random as random

__all__ = [
    "assert_equal",
    "assert_greater_equal",
    "function",
    "function_jit",
    "function_jit_fixedshape",
    "get_shape",
    "random",
    "tnp",
]

# Get shape dynamically (for graph mode compatibility)
get_shape = tf.shape
assert_equal = tf.assert_equal
assert_greater_equal = tf.debugging.assert_greater_equal

#: Whether to enable shape relaxation for JIT-compiled functions
RELAX_SHAPES = True

#: Standard TensorFlow function wrapper without JIT compilation
function = tf.function(autograph=False, jit_compile=False)

#: JIT-compiled TensorFlow function with shape relaxation enabled
function_jit = tf.function(
    autograph=False, reduce_retracing=RELAX_SHAPES, jit_compile=True
)

#: JIT-compiled TensorFlow function without shape relaxation
function_jit_fixedshape = tf.function(
    autograph=False, reduce_retracing=False, jit_compile=True
)

# Set eager mode based on environment variable
is_eager = bool(os.environ.get("PHASESPACE_EAGER"))
tf.config.run_functions_eagerly(is_eager)
