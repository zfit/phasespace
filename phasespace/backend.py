import tensorflow as tf

RELAX_SHAPES = True
function = tf.function(
    autograph=False, experimental_relax_shapes=RELAX_SHAPES, jit_compile=False
)
function_jit = tf.function(
    autograph=False, experimental_relax_shapes=RELAX_SHAPES, jit_compile=True
)
