import tensorflow as tf

RELAX_SHAPES = True
function = tf.function(autograph=False, jit_compile=False)
function_jit = tf.function(
    autograph=False, reduce_retracing=RELAX_SHAPES, jit_compile=True
)

function_jit_fixedshape = tf.function(
    autograph=False, reduce_retracing=False, jit_compile=True
)
