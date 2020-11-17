#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   kinematics.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   12.02.2019
# =============================================================================
"""Basic kinematics."""

import tensorflow as tf

RELAX_SHAPES = True


@tf.function(autograph=False, experimental_relax_shapes=RELAX_SHAPES)
def scalar_product(vec1, vec2):
    """Calculate scalar product of two 3-vectors.

    Arguments:
        vec1: First vector.
        vec2: Second vector.

    """
    return tf.reduce_sum(input_tensor=vec1 * vec2, axis=1)


@tf.function(autograph=False, experimental_relax_shapes=RELAX_SHAPES)
def spatial_component(vector):
    """Extract spatial components of the input Lorentz vector.

    Arguments:
        vector: Input Lorentz vector (where indexes 0-2 are space, index 3 is time).

    """
    return tf.gather(vector, indices=[0, 1, 2], axis=-1)


@tf.function(autograph=False, experimental_relax_shapes=RELAX_SHAPES)
def time_component(vector):
    """Extract time component of the input Lorentz vector.

    Arguments:
        vector: Input Lorentz vector (where indexes 0-2 are space, index 3 is time).

    """
    return tf.gather(vector, indices=[3], axis=-1)


@tf.function(autograph=False, experimental_relax_shapes=RELAX_SHAPES)
def x_component(vector):
    """Extract spatial X component of the input Lorentz or 3-vector.

    Arguments:
        vector: Input vector.

    """
    return tf.gather(vector, indices=[0], axis=-1)


@tf.function(autograph=False, experimental_relax_shapes=RELAX_SHAPES)
def y_component(vector):
    """Extract spatial Y component of the input Lorentz or 3-vector.

    Arguments:
        vector: Input vector.
    """
    return tf.gather(vector, indices=[1], axis=-1)


@tf.function(autograph=False, experimental_relax_shapes=RELAX_SHAPES)
def z_component(vector):
    """Extract spatial Z component of the input Lorentz or 3-vector.

    Arguments:
        vector: Input vector.

    """
    return tf.gather(vector, indices=[2], axis=-1)


@tf.function(autograph=False, experimental_relax_shapes=RELAX_SHAPES)
def mass(vector):
    """Calculate mass scalar for Lorentz 4-momentum.

    Arguments:
        vector: Input Lorentz momentum vector.

    """
    return tf.sqrt(
        tf.reduce_sum(
            input_tensor=tf.square(vector) * metric_tensor(), axis=-1, keepdims=True
        )
    )


@tf.function(autograph=False, experimental_relax_shapes=RELAX_SHAPES)
def lorentz_vector(space, time):
    """Make a Lorentz vector from spatial and time components.

    Arguments:
        space: 3-vector of spatial components.
        time: Time component.

    """
    return tf.concat([space, time], axis=-1)


@tf.function(autograph=False, experimental_relax_shapes=RELAX_SHAPES)
def lorentz_boost(vector, boostvector):
    """Perform Lorentz boost.

    Arguments:
        vector: 4-vector to be boosted
        boostvector: Boost vector. Can be either 3-vector or 4-vector, since only
            spatial components are used.

    """
    boost = spatial_component(boostvector)
    b2 = tf.expand_dims(scalar_product(boost, boost), axis=-1)

    def boost_fn():
        gamma = 1.0 / tf.sqrt(1.0 - b2)
        gamma2 = (gamma - 1.0) / b2
        ve = time_component(vector)
        vp = spatial_component(vector)
        bp = tf.expand_dims(scalar_product(vp, boost), axis=-1)
        vp2 = vp + (gamma2 * bp + gamma * ve) * boost
        ve2 = gamma * (ve + bp)
        return lorentz_vector(vp2, ve2)

    def no_boost_fn():
        return vector

    # if boost vector is zero, return the original vector
    all_b2_zero = tf.reduce_all(input_tensor=tf.equal(b2, tf.zeros_like(b2)))
    boosted_vector = tf.cond(pred=all_b2_zero, true_fn=no_boost_fn, false_fn=boost_fn)
    return boosted_vector


@tf.function(autograph=False, experimental_relax_shapes=RELAX_SHAPES)
def beta(vector):
    """Calculate beta of a given 4-vector.

    Arguments:
        vector: Input Lorentz momentum vector.

    """
    return mass(vector) / time_component(vector)


@tf.function(autograph=False, experimental_relax_shapes=RELAX_SHAPES)
def boost_components(vector):
    """Get the boost components of a given 4-vector.

    Arguments:
        vector: Input Lorentz momentum vector.

    """
    return spatial_component(vector) / time_component(vector)


@tf.function(autograph=False, experimental_relax_shapes=RELAX_SHAPES)
def metric_tensor():
    """Metric tensor for Lorentz space (constant)."""
    return tf.constant([-1.0, -1.0, -1.0, 1.0], dtype=tf.float64)

# EOF
