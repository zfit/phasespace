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
from typing import Union, Dict, Tuple, Optional, Callable

import tensorflow as tf

from . import kinematics as kin


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
    return tf.sqrt(x) / (tf.constant(2.0, dtype=tf.float64) * a)


class Particle:
    """Representation of a particle.

    Instances of this class can be combined with each other to build decay chains,
    which can then be used to generate phase space events through the `generate`
    or `generate_tensor` method.

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

    _sess_obj = None

    def __init__(self, name: str, mass: Union[Callable, int, float]) -> None:  # noqa
        self.name = name
        self.children = []
        if not callable(mass) and not isinstance(mass, tf.Variable):
            mass = tf.convert_to_tensor(mass, preferred_dtype=tf.float64)
            mass = tf.cast(mass, tf.float64)
        self._mass = mass
        self._n_events_var = None
        self._cache = None

    @property
    def _sess(self):
        """tf.Session: Internal session object."""
        sess = self._sess_obj
        if sess is None:
            sess = tf.Session()
            self._sess_obj = sess
        return sess

    @property
    def _n_events(self):
        """tf.Variable: Number of events to generate."""
        n_events_var = self._n_events_var
        if n_events_var is None:
            n_events_var = tf.Variable(initial_value=-42, dtype=tf.int64, use_resource=True, trainable=False)
            self._n_events_var = n_events_var
        return n_events_var

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

    def get_mass(self, min_mass: tf.Tensor = None, max_mass: tf.Tensor = None,
                 n_events: Union[tf.Tensor, tf.Variable] = None) -> tf.Tensor:
        """Get the particle mass.

        If the particle is resonant, the mass function will be called with the
        `min_mass`, `max_mass` and `n_events` parameters.

        Arguments:
            min_mass (`tf.Tensor`): Lower mass range. Defaults to None, which
                is only valid in the case of fixed mass.
            max_mass (`tf.Tensor`): Upper mass range. Defaults to None, which
                is only valid in the case of fixed mass.
            n_events (`tf.Tensor`): Number of events to produce. Has to be specified if the particle is resonant.

        Return:
            `tf.Tensor`: Mass.

        Raise:
            ValueError: If the mass is requested and has not been set.

        """
        if self.has_fixed_mass:
            mass = self._mass
        else:
            min_mass = tf.reshape(min_mass, (n_events,))
            max_mass = tf.reshape(max_mass, (n_events,))
            mass = self._mass(min_mass, max_mass, n_events)
        return mass

    @property
    def has_fixed_mass(self):
        """bool: Is the mass a callable function?"""
        return not callable(self._mass)

    @property
    def _cache_valid(self):
        return self._own_cache_valid and all(child._cache_valid for child in self.children)

    def _set_cache_validity(self, valid, propagate=False):
        self._own_cache_valid = valid
        if propagate:
            for child in self.children:
                child._set_cache_validity(valid, propagate=propagate)

    def set_children(self, *children):
        """Assign children.

        Arguments:
            children (list[Particle]): Children to assign to the current particle.

        Return:
            self

        Raise:
            ValueError: If there is an inconsistency in the parent/children relationship, ie,
            if children were already set, if their parent was or if less than two children were given.
            KeyError: If there is a particle name clash.

        """
        self._set_cache_validity(False)
        if self.children:
            raise ValueError("Children already set!")
        if len(children) <= 1:
            raise ValueError(f"Have to set at least 2 children, not {len(children)} for a particle to decay")
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
                momentum_shape = momentum.shape[0].value
                if momentum_shape is None:
                    momentum_shape = tf.shape(momentum, out_type=tf.int64)[0]
                else:
                    momentum_shape = tf.convert_to_tensor(momentum_shape, preferred_dtype=tf.int64)
                    momentum_shape = tf.cast(momentum_shape, dtype=tf.int64)
                assert_op = tf.assert_equal(n_events, momentum_shape,
                                            message="Conflicting inputs -> momentum_shape and n_events")
                with tf.control_dependencies([assert_op]):
                    n_events = tf.identity(n_events)
        if n_events is None:
            if momentum.shape.ndims == 2:
                n_events = momentum.shape[0].value
                if n_events is None:  # dynamic shape
                    n_events = tf.shape(momentum, out_type=tf.int64)[0]
            else:
                n_events = tf.constant(1, dtype=tf.int64)
        n_events = tf.convert_to_tensor(n_events, preferred_dtype=tf.int64)
        n_events = tf.cast(n_events, dtype=tf.int64)
        # Now preparation of tensors
        if momentum.shape.ndims == 1:
            momentum = tf.expand_dims(momentum, axis=0)
        return momentum, n_events

    @staticmethod
    def _get_w_max(available_mass, masses):
        emmax = available_mass + tf.gather(masses, indices=[0], axis=1)
        emmin = tf.zeros_like(emmax, dtype=tf.float64)
        w_max = tf.ones_like(emmax, dtype=tf.float64)
        for i in range(1, masses.shape[1].value):
            emmin += tf.gather(masses, [i - 1], axis=1)
            emmax += tf.gather(masses, [i], axis=1)
            w_max *= pdk(emmax, emmin, tf.gather(masses, [i], axis=1))
        return w_max

    def _generate(self, momentum, n_events):
        if not self.children:
            raise ValueError("No children have been configured")
        p_top, n_events = self._preprocess(momentum, n_events)
        top_mass = tf.broadcast_to(kin.mass(p_top), (n_events, 1))
        n_particles = len(self.children)

        # Prepare masses
        def recurse_stable(part):
            output_mass = 0
            for child in part.children:
                if child.has_fixed_mass:
                    output_mass += child.get_mass()
                else:
                    output_mass += recurse_stable(child)
            return output_mass

        mass_from_stable = tf.broadcast_to(
            tf.reduce_sum([child.get_mass() for child in self.children
                           if child.has_fixed_mass],
                          axis=0),
            (n_events, 1))
        max_mass = top_mass - mass_from_stable
        masses = []
        for child in self.children:
            if child.has_fixed_mass:
                masses.append(tf.broadcast_to(child.get_mass(), (n_events, 1)))
            else:
                # Recurse that particle to know the minimum mass we need to generate
                min_mass = tf.broadcast_to(recurse_stable(child), (n_events, 1))
                mass = child.get_mass(min_mass, max_mass, n_events)
                max_mass -= mass
                masses.append(tf.reshape(mass, (n_events, 1)))
        masses = tf.concat(masses, axis=-1)
        # if masses.shape.ndims == 1:
        #     masses = tf.expand_dims(masses, axis=0)
        available_mass = top_mass - tf.reduce_sum(masses, axis=1, keepdims=True)
        mass_check = tf.assert_greater_equal(available_mass, tf.zeros_like(available_mass, dtype=tf.float64),
                                             message="Forbidden decay",
                                             name="mass_check")
        with tf.control_dependencies([mass_check]):
            available_mass = tf.identity(available_mass)
        # Calculate the max weight, initial beta, etc
        w_max = self._get_w_max(available_mass, masses)
        p_top_boost = kin.boost_components(p_top)
        # Start the generation
        random_numbers = tf.random.uniform((n_events, n_particles - 2), dtype=tf.float64)
        random = tf.concat([tf.zeros((n_events, 1), dtype=tf.float64),
                            tf.sort(random_numbers, axis=1),
                            tf.ones((n_events, 1), dtype=tf.float64)],
                           axis=1)
        if random.shape[1].value is None:
            random.set_shape((None, n_particles))
        # random = tf.expand_dims(random, axis=-1)
        sum_ = tf.zeros((n_events, 1), dtype=tf.float64)
        inv_masses = []
        # TODO(Mayou36): rewrite with cumsum?
        for i in range(n_particles):
            sum_ += tf.gather(masses, [i], axis=1)
            inv_masses.append(tf.gather(random, [i], axis=1) * available_mass + sum_)
        pds = []
        # Calculate weights of the events
        for i in range(n_particles - 1):
            pds.append(pdk(inv_masses[i + 1], inv_masses[i], tf.gather(masses, [i + 1], axis=1)))
        weights = tf.reduce_prod(pds, axis=0)
        zero_component = tf.zeros_like(pds[0], dtype=tf.float64)
        generated_particles = [tf.concat([zero_component,
                                          pds[0],
                                          zero_component,
                                          tf.sqrt(tf.square(pds[0]) + tf.square(tf.gather(masses, [0], axis=1)))],
                                         axis=1)]
        part_num = 1
        while True:
            generated_particles.append(tf.concat([zero_component,
                                                  -pds[part_num - 1],
                                                  zero_component,
                                                  tf.sqrt(tf.square(pds[part_num - 1]) +
                                                          tf.square(tf.gather(masses, [part_num], axis=1)))],
                                                 axis=1))
            with tf.control_dependencies([n_events]):
                cos_z = (tf.constant(2.0, dtype=tf.float64) * tf.random.uniform((n_events, 1), dtype=tf.float64)
                         - tf.constant(1.0, dtype=tf.float64))
                sin_z = tf.sqrt(tf.constant(1.0, dtype=tf.float64) - cos_z * cos_z)
                ang_y = (tf.constant(2.0, dtype=tf.float64) * tf.constant(pi, dtype=tf.float64)
                         * tf.random.uniform((n_events, 1), dtype=tf.float64))
            cos_y = tf.math.cos(ang_y)
            sin_y = tf.math.sin(ang_y)
            # Do the rotations
            for j in range(part_num + 1):
                px = kin.x_component(generated_particles[j])
                py = kin.y_component(generated_particles[j])
                # Rotate about z
                # TODO(Mayou36): only list? will be overwritten below anyway, but can `*_component` handle it?
                generated_particles[j] = tf.concat([cos_z * px - sin_z * py,
                                                    sin_z * px + cos_z * py,
                                                    kin.z_component(generated_particles[j]),
                                                    kin.time_component(generated_particles[j])],
                                                   axis=1)
                # Rotate about y
                px = kin.x_component(generated_particles[j])
                pz = kin.z_component(generated_particles[j])
                generated_particles[j] = tf.concat([cos_y * px - sin_y * pz,
                                                    kin.y_component(generated_particles[j]),
                                                    sin_y * px + cos_y * pz,
                                                    kin.time_component(generated_particles[j])],
                                                   axis=1)
            if part_num == (n_particles - 1):
                break
            betas = (pds[part_num] / tf.sqrt(tf.square(pds[part_num]) + tf.square(inv_masses[part_num])))
            generated_particles = [kin.lorentz_boost(part,
                                                     tf.concat([zero_component,
                                                                betas,
                                                                zero_component],
                                                               axis=1))
                                   for part in generated_particles]
            part_num += 1
        # Final boost of all particles
        generated_particles = [kin.lorentz_boost(part, p_top_boost)
                               for part in generated_particles]
        return tf.reshape(weights, (n_events,)), tf.reshape(w_max, (n_events,)), generated_particles, masses

    def _recursive_generate(self, n_events, boost_to=None, recalculate_max_weights=False):
        if boost_to is not None:
            momentum = boost_to
        else:
            if self.has_fixed_mass:
                momentum = tf.broadcast_to(tf.stack((0.0, 0.0, 0.0, self.get_mass()), axis=-1), (n_events, 4))
            else:
                raise ValueError("Cannot use resonance as top particle")
        weights, weights_max, parts, children_masses = self._generate(momentum, n_events)
        output_particles = {child.name: parts[child_num]
                            for child_num, child in enumerate(self.children)}
        output_masses = {child.name: tf.gather(children_masses, [child_num], axis=1)
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
                w_max = tf.ones_like(available_mass)
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
                masses = tf.concat(masses, axis=1)
                w_max *= self._get_w_max(available_mass, masses)
                return w_max

            mass_tree = {}
            build_mass_tree(self, mass_tree)
            momentum = process_list_to_tensor(momentum)
            if len(momentum.shape.as_list()) == 1:
                momentum = tf.expand_dims(momentum, axis=-1)
            weights_max = tf.reshape(recurse_w_max(kin.mass(momentum), mass_tree[self.name]),
                                     (n_events,))
        return weights, weights_max, output_particles, output_masses

    def generate_tensor(self, n_events: Union[int, tf.Tensor, tf.Variable],
                        boost_to: Optional[tf.Tensor] = None,
                        normalize_weights: bool = True) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Generate normalized n-body phase space as tensorflow tensors.

        Events are generated in the rest frame of the particle, unless `boost_to` is given.

        Note:
            In this method, the event weights are returned normalized to their maximum.

        Arguments:
            n_events (int): Number of events to generate.
            boost_to (optional): Momentum vector of shape (x, 4), where x is optional, to where
                the resulting events will be boosted. If not specified, events are generated
                in the rest frame of the particle.
            normalize_weights (bool, optional): Normalize the event weight to its max?

        Return:
            tuple: Result of the generation, which varies with the value of `normalize_weights`:

                + If True, the tuple elements are the normalized event weights as a tensor of shape
                (n_events, ), and the momenta generated particles as a dictionary of tensors of shape
                (4, n_events) with particle names as keys.

                + If False, the tuple weights are the unnormalized event weights as a tensor of shape
                (n_events, ), the maximum per-event weights as a tensor of shape (n_events, ) and the
                momenta generated particles as a dictionary of tensors of shape (4, n_events) with particle
                names as keys.

        Raise:
            tf.errors.InvalidArgumentError: If the the decay is kinematically forbidden.
            ValueError: If `n_events` and the size of `boost_to` don't match. See `Particle.generate_unnormalized`.

        """
        if boost_to is not None:
            message = (f"The number of events requested ({n_events}) doesn't match the boost_to input size "
                       f"of {boost_to.shape}")
            assert_op = tf.assert_equal(tf.shape(boost_to)[0], tf.shape(n_events), message=message)
            with tf.control_dependencies([assert_op]):
                boost_to = tf.identity(boost_to)

        if not isinstance(n_events, tf.Variable):
            n_events = tf.convert_to_tensor(n_events, preferred_dtype=tf.int64)
            n_events = tf.cast(n_events, dtype=tf.int64)
        weights, weights_max, parts, _ = self._recursive_generate(n_events=n_events,
                                                                  boost_to=boost_to,
                                                                  recalculate_max_weights=self.has_grandchildren)
        return (weights / weights_max, parts) if normalize_weights else (weights, weights_max, parts)

    def generate(self, n_events: int, boost_to=None, normalize_weights: bool = True):
        """Generate normalized n-body phase space as numpy arrays.

        Events are generated in the rest frame of the particle, unless `boost_to` is given.

        Note:
            In this method, the event weights are returned normalized to their maximum.

        Arguments:
            n_events (int): Number of events to generate.
            boost_to (optional): Momentum vector of shape (x, 4), where x is optional, to where
                the resulting events will be boosted. If not specified, events are generated
                in the rest frame of the particle.
            normalize_weights (bool, optional): Normalize the event weight to its max?

        Return:
            tuple: Result of the generation, which varies with the value of `normalize_weights`:

                + If True, the tuple elements are the normalized event weights as an array of shape
                (n_events, ), and the momenta generated particles as a dictionary of arrays of shape
                (4, n_events) with particle names as keys.

                + If False, the tuple weights are the unnormalized event weights as an array of shape
                (n_events, ), the maximum per-event weights as an array of shape (n_events, ) and the
                momenta generated particles as a dictionary of arrays of shape (4, n_events) with particle
                names as keys.

        Raise:
            tf.errors.InvalidArgumentError: If the the decay is kinematically forbidden.
            ValueError: If `n_events` and the size of `boost_to` don't match. See `Particle.generate_unnormalized`.

        """
        # Convert n_events to a tf.Variable to perform graph caching
        if isinstance(n_events, tf.Variable):
            n_events_var = n_events
        else:
            if isinstance(n_events, tf.Tensor):
                raise TypeError("Tensor currently not allowed for generate. Use Python integers or `tf.Variable`.")
            n_events_var = self._n_events
            n_events_var.load(n_events, session=self._sess)
        # Run generation
        generate_tf = self._cache
        if generate_tf is None or not self._cache_valid or boost_to is not None:
            generate_tf = self.generate_tensor(n_events_var, boost_to, normalize_weights)
            self._cache = generate_tf
            self._set_cache_validity(True, propagate=True)
        return self._sess.run(generate_tf)


def generate_decay(mass_top: float, masses: list, n_events: Union[int, tf.Variable], boost_to=None,
                   as_numpy: bool = True):
    """Generate an n-body phasespace.

    Internally, this function uses `Particle` with a single generation of children.

    Note:
        This function doesn't cache so it may be slower on repeated calls. In that case
        it's better to use :py:class:`Particle` directly.

    Arguments:
        mass_top (`tf.Tensor`, list): Mass of the top particle. Can be a list of 4-vectors.
        masses (list): Masses of the child particles.
        n_events (int): Number of samples to generate.
        boost_to (`tf.Tensor` or array, optional): Momenta to boost the generated events to.
            If not specified, events are generated in the rest frame of the top particle.
        as_numpy(bool, optional): Return numpy arrays instead of tensorflow tensors? Defaults
            to True.

    Return:
        Tensor: 4-momenta of the generated particles, with shape (4xn_particles, n_events).
        ValueError: If nor `n_events` nor `boost_to` are given.

    """
    top = Particle('top', mass_top).set_children(*[Particle(str(num + 1), mass=mass)
                                                   for num, mass in enumerate(masses)])
    norm_weights, parts = top.generate(n_events=n_events,
                                       boost_to=boost_to,
                                       normalize_weights=True) if as_numpy \
        else top.generate_tensor(n_events=n_events, boost_to=boost_to, normalize_weights=True)
    return norm_weights, [parts[child.name] for child in top.children]

# EOF
