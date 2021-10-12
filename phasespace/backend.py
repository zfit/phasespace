import tensorflow as tf

RELAX_SHAPES = True
if int(tf.__version__.split(".")[1]) < 5:  # smaller than 2.5
    jit_compile_argname = "experimental_compile"
else:
    jit_compile_argname = "jit_compile"
function = tf.function(
    autograph=False,
    experimental_relax_shapes=RELAX_SHAPES,
    **{jit_compile_argname: False}
)
function_jit = tf.function(
    autograph=False,
    experimental_relax_shapes=RELAX_SHAPES,
    **{jit_compile_argname: True}
)

function_jit_fixedshape = tf.function(
    autograph=False, experimental_relax_shapes=False, **{jit_compile_argname: True}
)
