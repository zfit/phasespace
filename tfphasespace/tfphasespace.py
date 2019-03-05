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

# Useful constants
_ZERO = tf.constant(0.0, dtype=tf.float64)
_ONE = tf.constant(1.0, dtype=tf.float64)
_TWO = tf.constant(2.0, dtype=tf.float64)


def process_list_to_tensor(lst):
    if isinstance(lst, list):
        lst = tf.transpose(tf.convert_to_tensor(lst,
                                                preferred_dtype=tf.float64))
    return tf.cast(lst, dtype=tf.float64)


def pdk(a, b, c):
    """Calculate the PDK (2-body phase space) function."""
    x = (a - b - c) * (a + b + c) * (a - b + c) * (a + b - c)
    return tf.sqrt(x)/(_TWO * a)


class Particle:
    def __init__(self, name, parent=None):
        self.name = name
        self._children_masses = None
        self.parent = parent
        self.children = None
        self._n_events = None

    def _do_names_clash(self, names):
        if any(name == self.name for name in names):
            return self.name
        if not self.is_top():
            return self.parent._do_names_clash(names)
        return None

    def set_children(self, names, masses):
        # Check name clashes
        name_clash = self._do_names_clash(names)
        if name_clash:
            raise KeyError("Particle name {} already used".format(name_clash))
        # If none, add
        self._children_masses = masses
        self.children = [Particle(name, parent=self) for name in names]
        return self.children

    def is_top(self):
        return self.parent is None

    def has_children(self):
        return bool(self.children)

    def _preprocess(self, momentum, masses, n_events):
        momentum = process_list_to_tensor(momentum)
        masses = process_list_to_tensor(masses)
        momentum_shape = momentum.shape.as_list()
        mass_shape = masses.shape.as_list()
        # Check sanity of inputs
        if len(momentum_shape) not in (1, 2):
            raise ValueError("Bad shape for momentum -> {}".format(momentum_shape))
        if len(mass_shape) not in (1, 2):
            raise ValueError("Bad shape for masses -> {}".format(mass_shape))
        # Check compatibility of inputs
        if len(mass_shape) == 2:
            if n_events and n_events != mass_shape[1]:
                raise ValueError("Conflicting inputs -> masses and n_events")
            if len(momentum_shape) == 2:
                if mass_shape[1] != momentum_shape[1]:
                    raise ValueError("Conflicting inputs -> momentum and masses")
        if len(momentum_shape) == 2:
            if n_events and n_events != momentum_shape[1]:
                raise ValueError("Conflicting inputs -> momentum_shape and n_events")
        if not n_events:
            if len(mass_shape) == 2:
                n_events = mass_shape[1]
            elif len(momentum_shape) == 2:
                n_events = momentum_shape[1]
            else:
                n_events = 1
        # Now preparation of tensors
        if len(mass_shape) == 1:
            masses = tf.expand_dims(masses, axis=-1)
        if len(momentum_shape) == 1:
            momentum = tf.expand_dims(momentum, axis=-1)
        # TODO: Convert n_events to tf.Variable
        return momentum, masses, n_events

    def _generate(self, momentum, n_events):
        if not self.children:
            raise ValueError("No children have been configured")
        if callable(self._children_masses):
            masses = self._children_masses()
        else:
            masses = self._children_masses
        p_top, masses, n_events = self._preprocess(momentum, masses, n_events)
        n_particles = masses.shape.as_list()[0]
        top_mass = kin.mass(p_top)
        available_mass = top_mass - tf.reduce_sum(masses, axis=0)
        mass_check = tf.assert_greater_equal(available_mass, _ZERO,
                                             message="Forbidden decay",
                                             name="mass_check")
        with tf.control_dependencies([mass_check]):
            available_mass = tf.identity(available_mass)
        # Calculate the max weight, initial beta, etc
        emmax = available_mass + masses[0, :]
        emmin = _ZERO
        w_max = tf.ones((1, n_events), dtype=tf.float64)
        for i in range(1, n_particles):
            emmin += masses[i-1, :]
            emmax += masses[i, :]
            w_max *= pdk(emmax, emmin, masses[i, :])
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
        weights = tf.reduce_prod(pds, axis=0)
        zero_component = tf.zeros_like(pds[0], dtype=tf.float64)
        generated_particles = [tf.concat([zero_component,
                                          pds[0],
                                          zero_component,
                                          tf.sqrt(pds[0] * pds[0] + masses_unstacked[0] * masses_unstacked[0])],
                                         axis=0)]
        part_num = 1
        while True:
            generated_particles.append(tf.concat([zero_component,
                                                  -pds[part_num-1],
                                                  zero_component,
                                                  tf.sqrt(pds[part_num-1] * pds[part_num-1]
                                                          + masses_unstacked[part_num] * masses_unstacked[part_num])],
                                                 axis=0))
            cos_z = _TWO * tf.random.uniform((1, n_events), dtype=tf.float64) - _ONE
            sin_z = tf.sqrt(_ONE - cos_z * cos_z)
            ang_y = _TWO * tf.constant(pi, dtype=tf.float64) * tf.random.uniform((1, n_events), dtype=tf.float64)
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
            betas = (pds[part_num] / tf.sqrt(pds[part_num] * pds[part_num] + inv_masses[part_num] * inv_masses[part_num]))
            generated_particles = [kin.lorentz_boost(part,
                                                     tf.concat([zero_component,
                                                                betas,
                                                                zero_component],
                                                               axis=0),
                                                     dim_axis=0)
                                for part in generated_particles]
            part_num += 1
        # Final boost of all particles
        generated_particles = [kin.lorentz_boost(part, p_top_boost, dim_axis=0)
                               for part in generated_particles]
        return tf.reshape(weights, (n_events,)), tf.reshape(w_max, (n_events,)), generated_particles

    def generate(self, momentum, n_events):
        weights, weights_max, parts = self._generate(momentum, n_events)
        output_particles = {child.name: parts[child_num]
                            for child_num, child in enumerate(self.children)}
        return weights/weights_max, output_particles


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
    top = Particle('top')
    n_parts = process_list_to_tensor(masses).shape.as_list()[0]
    children_names = [str(num+1) for num in range(n_parts)]
    top.set_children(children_names, masses)
    norm_weights, parts = top.generate(p_top, n_events)
    return norm_weights, [parts[name] for name in children_names]


# EOF
