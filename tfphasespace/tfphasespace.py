#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   tfphasespace.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   25.02.2019
# =============================================================================
"""Implementation of the Raubold and Lynch method to generate n-body events.

The code is based on the GENBOD function (W515 from CERNLIB), documented in
    F. James, Monte Carlo Phase Space, CERN 68-15 (1968)

"""

from math import pi

import tensorflow as tf

import tfphasespace.kinematics as kin


def debug_print(op):
    p_op = tf.print(op)
    with tf.control_dependencies([p_op]):
        op = tf.identity(op)
    return op


def generate(p_top, masses, n_events=None):
    """Generate an n-body phasespace.

    Arguments:
        p_top (tf.tensor, list): Momentum of the top particle. Can be a list of 4-vectors.
        masses (list): Masses of the child particles. Can be a tensor of (n_particles, n_events) shape.
        n_events (int, optional): Number of samples to generate. If n_events is None,
            the shape of `masses` is used.

    Return:
        tf.tensor: 4-momenta of the generated particles.

    """
    # Useful constants
    zero = tf.constant(0.0, dtype=tf.float64)
    one = tf.constant(1.0, dtype=tf.float64)
    two = tf.constant(2.0, dtype=tf.float64)

    def pdk(a, b, c):
        """Calculate the PDK function."""
        x = (a - b - c) * (a + b + c) * (a - b + c) * (a + b - c)
        return tf.sqrt(x)/(two * a)

    # Bookkeeping, preparation
    if isinstance(p_top, list):
        p_top = tf.transpose(tf.convert_to_tensor(p_top,
                                                  preferred_dtype=tf.float64))
    p_top = tf.cast(p_top, dtype=tf.float64)
    if isinstance(masses, list):
        masses = tf.transpose(tf.convert_to_tensor(masses,
                                                   preferred_dtype=tf.float64))
    masses = tf.cast(masses, dtype=tf.float64)
    p_top_shape = p_top.shape.as_list()
    mass_shape = masses.shape.as_list()
    # Check sanity of inputs
    if len(p_top_shape) not in (1, 2):
        raise ValueError("Bad shape for p_top -> {}".format(p_top_shape))
    if len(mass_shape) not in (1, 2):
        raise ValueError("Bad shape for masses -> {}".format(mass_shape))
    # Check compatibility of inputs
    if len(mass_shape) == 2:
        if n_events and n_events != mass_shape[1]:
            raise ValueError("Conflicting inputs -> masses and n_events")
        if len(p_top_shape) == 2:
            if mass_shape[1] != p_top_shape[1]:
                raise ValueError("Conflicting inputs -> p_top and masses")
    if len(p_top_shape) == 2:
        if n_events and n_events != p_top_shape[1]:
            raise ValueError("Conflicting inputs -> p_top_shape and n_events")
    if not n_events:
        if len(mass_shape) == 2:
            n_events = mass_shape[1]
        elif len(p_top_shape) == 2:
            n_events = p_top_shape[1]
        else:
            n_events = 1
    # Now preparation of tensors
    n_particles = mass_shape[0]
    if len(mass_shape) == 1:
        masses = tf.expand_dims(masses, axis=-1)
    if len(p_top_shape) == 1:
        p_top = tf.expand_dims(p_top, axis=-1)
    # Check masses
    top_mass = kin.mass(p_top)
    available_mass = top_mass - tf.reduce_sum(masses, axis=0)
    mass_check = tf.assert_greater_equal(available_mass, zero,
                                         message="Forbidden decay",
                                         name="mass_check")
    with tf.control_dependencies([mass_check]):
        available_mass = tf.identity(available_mass)
    # Calculate the max weight, initial beta, etc
    emmax = available_mass + masses[0, :]
    emmin = zero
    w_max = tf.ones((1, n_events), dtype=tf.float64)
    for i in range(1, n_particles):
        emmin += masses[i-1, :]
        emmax += masses[i, :]
        w_max *= pdk(emmax, emmin, masses[i, :])
    w_max = one / w_max
    p_top_boost = kin.boost_components(p_top, axis=0)
    # Start the generation
    random_numbers = tf.random.uniform((n_particles-2, n_events), dtype=tf.float64)
    random = tf.concat([tf.zeros((1, n_events), dtype=tf.float64),
                        tf.contrib.framework.sort(random_numbers, axis=0),
                        tf.ones((1, n_events), dtype=tf.float64)],
                       axis=0)
    sum_ = tf.zeros((1, n_events), dtype=tf.float64)
    inv_masses = []
    masses_unstacked = tf.unstack(masses, axis=0)
    for i in range(n_particles):
        sum_ += masses_unstacked[i]
        inv_masses.append(random[i] * available_mass + sum_)
    pds = []
    # Calculate weights of the events
    for i in range(n_particles-1):
        pds.append(pdk(inv_masses[i+1], inv_masses[i], masses_unstacked[i+1]))
    weights = w_max * tf.reduce_prod(pds, axis=0)
    generated_particles = [tf.concat([tf.zeros((1, n_events), dtype=tf.float64),
                                      pds[0],
                                      tf.zeros((1, n_events), dtype=tf.float64),
                                      tf.sqrt(pds[0] * pds[0] + masses_unstacked[0] * masses_unstacked[0])],
                                     axis=0)]
    part_num = 1
    while True:
        generated_particles.append(tf.concat([tf.zeros((1, n_events), dtype=tf.float64),
                                              -pds[part_num-1],
                                              tf.zeros((1, n_events), dtype=tf.float64),
                                              tf.sqrt(pds[part_num-1] * pds[part_num-1] + masses_unstacked[part_num] * masses_unstacked[part_num])],
                                             axis=0))
        cos_z = two * tf.random.uniform((1, n_events), dtype=tf.float64) - one
        sin_z = tf.sqrt(one - cos_z * cos_z)
        ang_y = two * tf.constant(pi, dtype=tf.float64) * tf.random.uniform((1, n_events), dtype=tf.float64)
        cos_y = tf.math.cos(ang_y)
        sin_y = tf.math.sin(ang_y)
        # Do the rotations
        for j in range(part_num + 1):
            px = kin.x_component(generated_particles[j], axis=0)
            py = kin.y_component(generated_particles[j], axis=0)
            # Rotate about z
            generated_particles[j] = tf.concat([cos_z * px - sin_z * py,
                                                sin_z * px + cos_z * py,
                                                tf.expand_dims(kin.z_component(generated_particles[j], axis=0),
                                                               axis=0),
                                                tf.expand_dims(kin.time_component(generated_particles[j], axis=0),
                                                               axis=0)],
                                               axis=0)
            # Rotate about y
            px = kin.x_component(generated_particles[j], axis=0)
            pz = kin.z_component(generated_particles[j], axis=0)
            generated_particles[j] = tf.concat([cos_y * px - sin_y * pz,
                                                tf.expand_dims(kin.y_component(generated_particles[j], axis=0),
                                                               axis=0),
                                                sin_y * px + cos_y * pz,
                                                tf.expand_dims(kin.time_component(generated_particles[j], axis=0),
                                                               axis=0)],
                                               axis=0)
        if part_num == (n_particles - 1):
            break
        betas = (pds[i] / tf.sqrt(pds[i] * pds[i] + inv_masses[i] * inv_masses[i]))
        generated_particles = [kin.lorentz_boost(part,
                                                 tf.concat([tf.zeros_like(betas),
                                                            betas,
                                                            tf.zeros_like(betas)],
                                                           axis=0),
                                                 dim_axis=0)
                               for part in generated_particles]
        part_num += 1
    # Final boost of all particles
    generated_particles = [kin.lorentz_boost(part, p_top_boost, dim_axis=0)
                           for part in generated_particles]
    # Should we merge the particles?
    # tf.concat(generated_particles, axis=0))
    return tf.reshape(weights, (n_events,)), generated_particles


# EOF

