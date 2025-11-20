"""TensorFlow backend function decorators for phasespace.

This module provides standardized TensorFlow function decorators for different compilation and optimization
strategies used throughout the phasespace package.
"""

import tensorflow as tf

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
