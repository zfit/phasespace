#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   kinematics.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   12.02.2019
# =============================================================================
"""Basic kinematics."""

import tensorflow as tf


def scalar_product(vec1, vec2):
    """
    Calculate scalar product of two 3-vectors
    """
    return tf.reduce_sum(vec1 * vec2, axis=1)


def spatial_component(vector):
    """
    Return spatial components of the input Lorentz vector
        vector : input Lorentz vector (where indexes 0-2 are space, index 3 is time)
    """
    return tf.transpose(tf.stack(tf.unstack(vector, axis=1)[0:3]))


def time_component(vector):
    """
    Return time component of the input Lorentz vector
        vector : input Lorentz vector (where indexes 0-2 are space, index 3 is time)
    """
    return tf.expand_dims(tf.unstack(vector, axis=1)[3], axis=-1)


def x_component(vector):
    """
    Return spatial X component of the input Lorentz or 3-vector
        vector : input vector
    """
    return tf.expand_dims(tf.unstack(vector, axis=1)[0], axis=-1)


def y_component(vector):
    """
    Return spatial Y component of the input Lorentz or 3-vector
        vector : input vector
    """
    return tf.expand_dims(tf.unstack(vector, axis=1)[1], axis=-1)


def z_component(vector):
    """
    Return spatial Z component of the input Lorentz or 3-vector
        vector : input vector
    """
    return tf.expand_dims(tf.unstack(vector, axis=1)[2], axis=-1)


def mass(vector):
    """
    Calculate mass scalar for Lorentz 4-momentum
        vector : input Lorentz momentum vector
    """
    return tf.expand_dims(
        tf.sqrt(tf.reduce_sum(tf.square(vector) * metric_tensor(),
                              axis=1)),
        axis=-1)


def lorentz_vector(space, time):
    """
    Make a Lorentz vector from spatial and time components
        space : 3-vector of spatial components
        time  : time component
    """
    return tf.concat([space, time], axis=1)


def lorentz_boost(vector, boostvector):
    """
    Perform Lorentz boost
        vector :     4-vector to be boosted
        boostvector: boost vector. Can be either 3-vector or 4-vector (only spatial components
        are used)
    """
    boost = spatial_component(boostvector)
    b2 = tf.expand_dims(scalar_product(boost, boost), axis=-1)

    def boost_fn():
        gamma = 1. / tf.sqrt(1. - b2)
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
    all_b2_zero = tf.reduce_all(tf.equal(b2, tf.zeros_like(b2)))
    boosted_vector = tf.cond(all_b2_zero, true_fn=no_boost_fn, false_fn=boost_fn)
    return boosted_vector


def beta(vector, axis=1):
    """Calculate beta of a given 4-vector"""
    return mass(vector) / time_component(vector)


def boost_components(vector):
    """Get the boost components of a given vector."""
    return spatial_component(vector) / time_component(vector)


def metric_tensor():
    """
    Metric tensor for Lorentz space (constant)
    """
    return tf.constant([-1., -1., -1., 1.], dtype=tf.float64)

# EOF
