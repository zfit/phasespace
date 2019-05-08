#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   phasespace.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   25.02.2019
# =============================================================================
"""Implementation of the Raubold and Lynch method to generate n-body events.

The code is based on the GENBOD function (W515 from CERNLIB), documented in
    F. James, Monte Carlo Phase Space, CERN 68-15 (1968)

"""

from math import pi

import tensorflow as tf

import phasespace.kinematics as kin

# Useful constants
_ZERO = tf.constant(0.0, dtype=tf.float64)
_ONE = tf.constant(1.0, dtype=tf.float64)
_TWO = tf.constant(2.0, dtype=tf.float64)


def process_list_to_tensor(lst):
    """Convert a list to a tensor.

    The list is converted to a tensor and transposed to get the proper shape.

    Note:
        If `lst` is a tensor, nothing is done to it other than convert it to `tf.float64`.

    Arguments:
        lst (list): List to convert.

    Return:
        Tensor.

    """
    if isinstance(lst, list):
        lst = tf.transpose(tf.convert_to_tensor(lst,
                                                preferred_dtype=tf.float64))
    return tf.cast(lst, dtype=tf.float64)


def pdk(a, b, c):
    """Calculate the PDK (2-body phase space) function.

    Based on Eq. (9.17) in CERN 68-15 (1968).

    Arguments:
        a (Tensor): :math:`M_{i+1}` in Eq. (9.17).
        b (Tensor): :math:`M_{i}` in Eq. (9.17).
        c (Tensor): :math:`m_{i+1}` in Eq. (9.17).

    Return:
        Tensor.

    """
    x = (a - b - c) * (a + b + c) * (a - b + c) * (a + b - c)
    return tf.sqrt(x) / (_TWO * a)


class Particle:
    """Representation of a particle.

    Instances of this class can be combined with each other to build decay chains,
    which can then be used to generate phase space events through the `generate`
    method.

    A `Particle` must have a `name`, which is ensured not to clash with any others in
    the decay chain.
    It may also have:

        + Mass, which can be either a number or a function to generate it according to
            a certain distribution. In this case, the particle is not considered as having a
            fixed mass and the `has_fixed_mass` method will return False.
        + Children, ie, decay products, which are also `Particle` instances.


    Arguments:
        name (str): Name of the particle.
        mass (float, Tensor, callable): Mass of the particle. If it's a float, it get
            converted to a `tf.constant`.

    """

    def __init__(self, name, mass):  # noqa
        self.name = name
        self.children = []
        if not callable(mass) and not isinstance(mass, tf.Variable):
            mass = tf.convert_to_tensor(mass, preferred_dtype=tf.float64)
            mass = tf.cast(mass, tf.float64)
        self._mass = mass

    def _do_names_clash(self, particles):
        def get_list_of_names(part):
            output = [part.name]
            for child in part.children:
                output.extend(get_list_of_names(child))
            return output

        names_to_check = [self.name]
        for part in particles:
            names_to_check.extend(get_list_of_names(part))
        # Find top
        dup_names = {name for name in names_to_check if names_to_check.count(name) > 1}
        if dup_names:
            return dup_names
        return None

    def get_mass(self, min_mass=None, max_mass=None, n_events=None):
        """Get the particle mass.

        If the particle is resonant, the mass function will be called with the
        `min_mass` and `max_mass` parameters.

        Arguments:
            min_mass (tensor): Lower mass range. Defaults to None, which
                is only valid in the case of fixed mass.
            max_maxx (tensor): Upper mass range. Defaults to None, which
                is only valid in the case of fixed mass.

        Return:
            tensor: Mass.

        Raise:
            ValueError: If the mass is requested and has not been set.

        """
        if self._mass is None:
            raise ValueError("Mass has not been configured!")
        if self.has_fixed_mass:
            return self._mass
        else:
            return self._mass(min_mass, max_mass, n_events)

    @property
    def has_fixed_mass(self):
        """bool: Is the mass a callable function?"""
        return not callable(self._mass)

    def set_children(self, *children):
        """Assign children.

        Arguments:
            children (list[Particle]): Children to assign to the current particle.

        Return:
            self

        Raise:
            ValueError: If there is an inconsistency in the parent/children relationship, ie,
            if children were already set, or if their parent was.
            KeyError: If there is a particle name clash.

        """
        if self.children:
            raise ValueError("Children already set!")
        # Check name clashes
        name_clash = self._do_names_clash(children)
        if name_clash:
            raise KeyError("Particle name {} already used".format(name_clash))
        self.children = children
        return self

    @property
    def has_children(self):
        """bool: Does the particle have children?"""
        return bool(self.children)

    @property
    def has_grandchildren(self):
        """bool: Does the particle have grandchildren?"""
        if not self.children:
            return False
        return any(child.has_children for child in self.children)

    @staticmethod
    def _preprocess(momentum, n_events):
        """Preprocess momentum input and determine number of events to generate.

        Both `momentum` and `n_events` are converted to tensors if they
        are not already.

        Arguments:
            `momentum`: Momentum vector, of shape (4, x), where x is optional.
            `n_events`: Number of events to generate. If `None`, the number of events
            to generate is calculated from the shape of `momentum`.

        Return:
            tuple: Processed `momentum` and `n_events`.

        Raise:
            tf.errors.InvalidArgumentError: If the number of events deduced from the
            shape of `momentum` is inconsistent with `n_events`.

        """
        momentum = process_list_to_tensor(momentum)

        # Check sanity of inputs
        if momentum.shape.ndims not in (1, 2):
            raise ValueError("Bad shape for momentum -> {}".format(momentum.shape.as_list()))
        # Check compatibility of inputs
        if momentum.shape.ndims == 2:
            if n_events is not None:
                momentum_shape = momentum.shape[1].value
                if momentum_shape is None:
                    momentum_shape = tf.shape(momentum, out_type=tf.int32)[1]
                else:
                    momentum_shape = tf.convert_to_tensor(momentum_shape, preferred_dtype=tf.int32)
                    momentum_shape = tf.cast(momentum_shape, dtype=tf.int32)
                assert_op = tf.assert_equal(n_events, momentum_shape,
                                            message="Conflicting inputs -> momentum_shape and n_events")
                with tf.control_dependencies([assert_op]):
                    n_events = tf.identity(n_events)
        if n_events is None:
            if momentum.shape.ndims == 2:
                n_events = momentum.shape[1].value
                if n_events is None:  # dynamic shape
                    n_events = tf.shape(momentum, out_type=tf.int32)[1]
            else:
                n_events = tf.constant(1, dtype=tf.int32)
        n_events = tf.convert_to_tensor(n_events, preferred_dtype=tf.int32)
        n_events = tf.cast(n_events, dtype=tf.int32)
        # Now preparation of tensors
        if momentum.shape.ndims == 1:
            momentum = tf.expand_dims(momentum, axis=-1)
        return momentum, n_events

    def _get_w_max(self, available_mass, masses):
        emmax = available_mass + masses[0, :]
        emmin = _ZERO
        w_max = _ONE
        for i in range(1, masses.shape.as_list()[0]):
            emmin += masses[i - 1, :]
            emmax += masses[i, :]
            w_max *= pdk(emmax, emmin, masses[i, :])
        return w_max

    def _generate(self, momentum, n_events):
        if not self.children:
            raise ValueError("No children have been configured")
        p_top, n_events = self._preprocess(momentum, n_events)
        top_mass = kin.mass(p_top)
        n_particles = len(self.children)

        # Prepare masses
        def recurse_stable(part):
            output_mass = tf.zeros(tuple(), dtype=tf.float64)
            for child in part.children:
                if child.has_fixed_mass:
                    output_mass += child.get_mass()
                else:
                    output_mass += recurse_stable(child)
            return output_mass

        mass_from_stable = tf.reduce_sum([child.get_mass() for child in self.children
                                          if child.has_fixed_mass],
                                         axis=0)
        max_mass = top_mass - mass_from_stable
        masses = []
        for child in self.children:
            if child.has_fixed_mass:
                masses.append(tf.broadcast_to(child.get_mass(), (1, n_events)))
            else:
                # Recurse that particle to know the minimum mass we need to generate
                min_mass = tf.broadcast_to(recurse_stable(child), tf.shape(max_mass))
                mass = child.get_mass(min_mass, max_mass, n_events)
                max_mass -= mass
                masses.append(mass)
        masses = tf.concat(masses, axis=0)
        available_mass = top_mass - tf.reduce_sum(masses, axis=0)
        mass_check = tf.assert_greater_equal(available_mass, _ZERO,
                                             message="Forbidden decay",
                                             name="mass_check")
        with tf.control_dependencies([mass_check]):
            available_mass = tf.identity(available_mass)
        # Calculate the max weight, initial beta, etc
        w_max = self._get_w_max(available_mass, masses)
        p_top_boost = kin.boost_components(p_top, axis=0)
        # Start the generation
        random_numbers = tf.random.uniform((n_particles - 2, n_events), dtype=tf.float64)
        random = tf.concat([tf.zeros((1, n_events), dtype=tf.float64),
                            tf.contrib.framework.sort(random_numbers, axis=0),
                            tf.ones((1, n_events), dtype=tf.float64)],
                           axis=0)
        if random.shape[0] is None:
            random.set_shape((n_particles, None))
        sum_ = tf.zeros((1, n_events), dtype=tf.float64)
        inv_masses = []
        # TODO(Mayou36): rewrite with cumsum?
        for i in range(n_particles):
            sum_ += masses[i]
            inv_masses.append(random[i] * available_mass + sum_)
        pds = []
        # Calculate weights of the events
        for i in range(n_particles - 1):
            pds.append(pdk(inv_masses[i + 1], inv_masses[i], masses[i + 1]))
        weights = tf.reduce_prod(pds, axis=0)
        zero_component = tf.zeros_like(pds[0], dtype=tf.float64)
        generated_particles = [tf.concat([zero_component,
                                          pds[0],
                                          zero_component,
                                          tf.sqrt(tf.square(pds[0]) + tf.square(masses[0]))],
                                         axis=0)]
        part_num = 1
        while True:
            generated_particles.append(tf.concat([zero_component,
                                                  -pds[part_num - 1],
                                                  zero_component,
                                                  tf.sqrt(tf.square(pds[part_num - 1]) + tf.square(masses[part_num]))],
                                                 axis=0))

            with tf.control_dependencies([n_events]):
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
                # TODO(Mayou36): only list? will be overwritten below anyway, but can `*_component` handle it?
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
            betas = (pds[part_num] / tf.sqrt(tf.square(pds[part_num]) + tf.square(inv_masses[part_num])))
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
        return tf.reshape(weights, (n_events,)), w_max, generated_particles, masses

    def _recursive_generate(self, n_events, boost_to=None, recalculate_max_weights=False):
        if boost_to is not None:
            momentum = boost_to
        else:
            if self.has_fixed_mass:
                momentum = tf.stack((0.0, 0.0, 0.0, self.get_mass()), axis=-1)
            else:
                raise ValueError("Cannot use resonance as top particle")
        weights, weights_max, parts, children_masses = self._generate(momentum, n_events)
        output_particles = {child.name: parts[child_num]
                            for child_num, child in enumerate(self.children)}
        output_masses = {child.name: children_masses[child_num]
                         for child_num, child in enumerate(self.children)}
        for child_num, child in enumerate(self.children):
            if child.has_children:
                child_weights, _, child_gen_particles, child_masses = \
                    child._recursive_generate(n_events=n_events,
                                              boost_to=parts[child_num],
                                              recalculate_max_weights=False)
                weights *= child_weights
                output_particles.update(child_gen_particles)
                output_masses.update(child_masses)
        if recalculate_max_weights:

            def build_mass_tree(particle, leaf):
                if particle.has_children:
                    leaf[particle.name] = {}
                    for child in particle.children:
                        build_mass_tree(child, leaf[particle.name])
                else:
                    leaf[particle.name] = output_masses[particle.name]

            def get_flattened_values(dict_):
                output = []
                for val in dict_.values():
                    if isinstance(val, dict):
                        output.extend(get_flattened_values(val))
                    else:
                        output.append(val)
                return output

            def recurse_w_max(parent_mass, current_mass_tree):
                available_mass = parent_mass - sum(get_flattened_values(current_mass_tree))
                masses = []
                w_max = tf.expand_dims(_ONE, axis=-1)
                for child, child_mass in current_mass_tree.items():
                    if isinstance(child_mass, dict):
                        w_max *= recurse_w_max(parent_mass -
                                               sum(get_flattened_values({ch_it: ch_m_it
                                                                         for ch_it, ch_m_it
                                                                         in current_mass_tree.items()
                                                                         if ch_it != child})),
                                               child_mass)
                        masses.append(sum(get_flattened_values(child_mass)))
                    else:
                        masses.append(child_mass)
                # Find largest mass tensor

                masses_shape = tuple(mass.shape.as_list()[0] for mass in masses)
                if None in masses_shape:
                    masses_shape = tuple(tf.shape(mass) for mass in masses)
                    max_shape = tf.reduce_max(masses_shape)
                else:
                    max_shape = max(masses_shape)

                masses = tf.convert_to_tensor([tf.broadcast_to(mass, (max_shape,)) for mass in masses])
                if len(masses.shape.as_list()) == 1:
                    masses = tf.expand_dims(masses, axis=-1)
                w_max *= self._get_w_max(available_mass, masses)
                return w_max

            mass_tree = {}
            build_mass_tree(self, mass_tree)
            momentum = process_list_to_tensor(momentum)
            if len(momentum.shape.as_list()) == 1:
                momentum = tf.expand_dims(momentum, axis=-1)
            weights_max = tf.ones_like(weights, dtype=tf.float64) * \
                          recurse_w_max(kin.mass(momentum), mass_tree[self.name])
        return weights, weights_max, output_particles, output_masses

    def generate_unnormalized(self, n_events=None, boost_to=None):
        """Generate unnormalized n-body phase space.

        Note:
            In this method, the event weights and their normalization (the maximum weight)
            are returned separately.

        Note:
            If nor `n_events` nor `boost_to` is given, a single event is generated in the
            rest frame of the particle.

        Arguments:
            `n_events` (optional): Number of events to generate. If `None` (default),
            the number of events to generate is calculated from the shape of `boost`.
            `boost_to`: Momentum vector of shape (4, x), where x is optional, where
            the resulting events will be boosted to.

        Return:
            tuple: Event weights tensor of shape (n_events, ), max event weights tensor, of shape
            (n_events, ), and generated particles, a dictionary of tensors of shape (4, n_events)
            with particle names as keys.

        Raise:
            tf.errors.InvalidArgumentError: If the the decay is kinematically forbidden.

        """
        if n_events is None:
            if boost_to is None or boost_to.shape.ndims == 1:
                n_events = 1
            else:
                n_events = tf.shape(boost_momentum)[1]
        if not isinstance(n_events, tf.Variable):
            n_events = tf.convert_to_tensor(n_events, preferred_dtype=tf.int32)
            n_events = tf.cast(n_events, dtype=tf.int32)
        weights, weights_max, parts, _ = self._recursive_generate(n_events=n_events,
                                                                  boost_to=boost_to,
                                                                  recalculate_max_weights=self.has_grandchildren)
        return weights, weights_max, parts

    def generate(self, n_events=None, boost_to=None):
        """Generate normalized n-body phase space.

        Events are generated in the rest frame of the particle, unless `boost_to` is given.

        Note:
            In this method, the event weights are returned normalized to their maximum.

        Note:
            If nor `n_events` nor `boost_to` is given, a single event is generated in the
            rest frame of the particle.

        Arguments:
            `n_events` (optional): Number of events to generate. If `None` (default),
            the number of events to generate is calculated from the shape of `boost`.
            `boost_to`: Momentum vector of shape (4, x), where x is optional, where
            the resulting events will be boosted to.

        Return:
            tuple: Normalized event weights tensor of shape (n_events, ), and generated
            particles, a dictionary of tensors of shape (4, n_events) with particle names
            as keys.

        Raise:
            tf.errors.InvalidArgumentError: If the the decay is kinematically forbidden.

        """
        if n_events is None and boost_to is None:
            n_events = 1
        weights, weights_max, parts = self.generate_unnormalized(n_events, boost_to)
        return weights / weights_max, parts


def generate(mass_top, masses, n_events=None, boost_to=None):
    """Generate an n-body phasespace.

    Internally, this function uses `Particle` with a single generation of children.

    Arguments:
        p_top (Tensor, list): Momentum of the top particle. Can be a list of 4-vectors.
        masses (list): Masses of the child particles.
        n_events (int, optional): Number of samples to generate. If n_events is None,
            the number of events is deduced from `p_top`.

    Return:
        Tensor: 4-momenta of the generated particles, with shape (4xn_particles, n_events).

    """
    top = Particle('top', mass_top).set_children(*[Particle(str(num + 1), mass=mass)
                                                   for num, mass in enumerate(masses)])
    norm_weights, parts = top.generate(n_events=n_events, boost_to=boost_to)
    return norm_weights, [parts[child.name] for child in top.children]

# EOF
