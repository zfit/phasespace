#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   kinematics.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   12.02.2019
# =============================================================================
"""Basic kinematics."""

import tensorflow as tf


def scalar_product(vec1, vec2, axis=1):
    """
    Calculate scalar product of two 3-vectors
    """
    return tf.reduce_sum(vec1 * vec2, axis=axis)

def spatial_component(vector, axis=1):
    """
    Return spatial components of the input Lorentz vector
        vector : input Lorentz vector (where indexes 0-2 are space, index 3 is time)
    """
    return tf.stack(tf.unstack(vector, axis=axis)[0:3])
    # return vector[:, 0:3]


def time_component(vector, axis=1):
    """
    Return time component of the input Lorentz vector
        vector : input Lorentz vector (where indexes 0-2 are space, index 3 is time)
    """
    return tf.unstack(vector, axis=axis)[3]


def x_component(vector, axis=1):
    """
    Return spatial X component of the input Lorentz or 3-vector
        vector : input vector
    """
    return tf.unstack(vector, axis=axis)[0]


def y_component(vector, axis=1):
    """
    Return spatial Y component of the input Lorentz or 3-vector
        vector : input vector
    """
    return tf.unstack(vector, axis=axis)[1]


def z_component(vector, axis=1):
    """
    Return spatial Z component of the input Lorentz or 3-vector
        vector : input vector
    """
    return tf.unstack(vector, axis=axis)[2]


def vector(x, y, z):
    """
    Make a 3-vector from components
    x, y, z : vector components
    """
    return tf.stack([x, y, z], axis=0)


def scalar(x):
    """
    Create a scalar (e.g. tensor with only one component) which can be used to e.g. scale a vector
    One cannot do e.g. Const(2.)*vector(x, y, z), needs to do scalar(Const(2))*vector(x, y, z)
    """
    return tf.stack([x], axis=1)


def mass(vector):
    """
    Calculate mass scalar for Lorentz 4-momentum
        vector : input Lorentz momentum vector
    """
    return tf.sqrt(tf.reduce_sum(vector * vector * tf.reshape(metric_tensor(), (4,1)), axis=0))
    # return tf.sqrt(tf.reduce_sum(vector * vector * metric_tensor(), axis=0))


def lorentz_vector(space, time):
    """
    Make a Lorentz vector from spatial and time components
        space : 3-vector of spatial components
        time  : time component
    """
    return tf.concat([space, tf.stack([time], axis=0)], axis=0)


def lorentz_boost(vector, boostvector, dim_axis=1):
    """
    Perform Lorentz boost
        vector :     4-vector to be boosted
        boostvector: boost vector. Can be either 3-vector or 4-vector (only spatial components
        are used)
    """
    boost = spatial_component(boostvector, axis=dim_axis)
    b2 = scalar_product(boost, boost, dim_axis)
    gamma = 1. / tf.sqrt(1. - b2)
    gamma2 = (gamma - 1.0) / b2
    ve = time_component(vector, axis=dim_axis)
    vp = spatial_component(vector, axis=dim_axis)
    bp = scalar_product(vp, boost, dim_axis)
    vp2 = vp + (gamma2 * bp + gamma * ve) * boost
    ve2 = gamma * (ve + bp)
    return lorentz_vector(vp2, ve2)


def beta(vector, axis=1):
    """Calculate beta of a given 4-vector"""
    return mass(vector)/time_component(vector, axis=axis)


def boost_components(vector, axis=1):
    """Get the boost components of a given vector."""
    return spatial_component(vector, axis=axis)/time_component(vector, axis=axis)


def metric_tensor():
    """
    Metric tensor for Lorentz space (constant)
    """
    return tf.constant([-1., -1., -1., 1.], dtype=tf.float64)


def lorentz_dot_product(vec1, vec2):
    """
    Dot product of two lorentz vectors
    return tf.tensordot(vec1,vec2,)?
    """
    return tf.reduce_sum(vec1 * vec2 * metric_tensor(), axis=-1)

# EOF
