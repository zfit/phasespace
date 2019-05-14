#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   generator.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   14.05.2019
# =============================================================================
"""Phasespace generator."""

import tensorflow as tf


class PhasespaceGenerator:
    """Handle generation of decays.

    Takes care of caching and running the session such that numpy arrays are
    obtained. This allows for generating in chunks without recreating the
    graph.

    Example:
        Generate 100 events according to the `decay` decay chain.

        >>> gen = decay.get_generator()
        >>> weights, parts = gen.generate(n_events=100)

    Example:
        Generate 1000 events of the `decay` decay chain in chunks of 100.

        >>> gen = decay.get_generator()
        >>> particles = []
        >>> for _ in range(10):
        ...     weights, parts = gen.generate(n_events=100)
        ...     particles.extend(parts)

    Arguments:
        phasespace.Particle: Decay to handle.

    """

    def __init__(self, decay):  # noqa
        self.decay = decay
        self.sess = tf.Session()
        self._n_events = None

    def get_n_events(self, n_events):
        """Get n_events variable.

        Initialize the cached `tf.Variable` if necessary, and ssign its value too.

        Arguments:
            n_events (int): Number of events.

        Return:
            tf.Variable

        """
        if self._n_events is None:
            self._n_events = tf.Variable(initial_value=n_events,
                                         dtype=tf.int64, use_resource=True)
        else:
            self._n_events.load(n_events, session=self.sess)
        return self._n_events


    def generate(self, n_events=None, boost_to=None):
        """Generate normalized n-body phase space.

        Events are generated in the rest frame of the particle, unless `boost_to` is given.

        Note:
            In this method, the event weights are returned normalized to their maximum.

        Note:
            If nor `n_events` nor `boost_to` is given, a single event is generated in the
            rest frame of the particle.

        Arguments:
            n_events (optional): Number of events to generate. If `None` (default),
                the number of events to generate is calculated from the shape of `boost`.
            boost_to (optional): Momentum vector of shape (4, x), where x is optional, where
                the resulting events will be boosted to. If not specified, events are generated
                in the rest frame of the particle.

        Return:
            tuple: Normalized event weights array of shape (n_events, ), and generated
            particles, a dictionary of  arrays of shape (4, n_events) with particle names
            as keys.

        Raise:
            tf.errors.InvalidArgumentError: If the the decay is kinematically forbidden.

        """
        # Determine the number of events
        if n_events is None and boost_to is None:
            n_events = 1
        # Convert n_events to a tf.Variable to perform graph caching
        n_events = self.get_n_events(n_events)
        # Run generation
        return self.sess.run(self.decay.generate(n_events, boost_to))

# EOF
