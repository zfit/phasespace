#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   test_tfphasespace.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   27.02.2019
# =============================================================================
"""Basic dimensionality tests."""

import pytest

import tensorflow as tf

import tfphasespace

import os
import sys

sys.path.append(os.path.dirname(__file__))

from .helpers import decays

B_AT_REST = decays.B0_AT_REST
BS_AT_REST = tf.stack((0.0, 0.0, 0.0, decays.B0_MASS + 86.8), axis=-1)
PION_MASS = decays.PION_MASS


def test_one_event():
    """Test B->pi pi pi."""
    with tf.Session() as sess:
        norm_weights, particles = sess.run(tfphasespace.generate(B_AT_REST,
                                                                 [PION_MASS, PION_MASS, PION_MASS]))
        assert len(norm_weights) == 1
        assert all([weight < 1 for weight in norm_weights])
        assert len(particles) == 3
        assert all([part.shape == (4, 1) for part in particles])


@pytest.mark.parametrize("n_events", argvalues=[5, tf.constant(5), tf.Variable(initial_value=5)])
def test_n_events(n_events):
    """Test 5 B->pi pi pi."""
    with tf.Session() as sess:
        if isinstance(n_events, tf.Variable):
            sess.run(n_events.initializer)
        norm_weights, particles = sess.run(tfphasespace.generate(B_AT_REST,
                                                                 [PION_MASS, PION_MASS, PION_MASS],
                                                                 n_events=n_events))
        assert len(norm_weights) == 5
        assert all([weight < 1 for weight in norm_weights])
        assert len(particles) == 3
        assert all([part.shape == (4, 5) for part in particles])


def test_n_events_implicit_parent():
    """Test multiple events by passing mutiple parent momenta."""
    with tf.Session() as sess:
        norm_weights, particles = sess.run(tfphasespace.generate([B_AT_REST, BS_AT_REST],
                                                                 [PION_MASS, PION_MASS, PION_MASS]))
        assert len(norm_weights) == 2
        assert all([weight < 1 for weight in norm_weights])
        assert len(particles) == 3
        assert all([part.shape == (4, 2) for part in particles])


def test_n_events_implicit_daughters():
    """Test multiple events by passing mutiple daugther masses."""
    with tf.Session() as sess:
        norm_weights, particles = sess.run(tfphasespace.generate(B_AT_REST,
                                                                 [[PION_MASS, PION_MASS, PION_MASS],
                                                                  [PION_MASS, PION_MASS, PION_MASS]]))
        assert len(norm_weights) == 2
        assert all([weight < 1 for weight in norm_weights])
        assert len(particles) == 3
        assert all([part.shape == (4, 2) for part in particles])


def test_input_inconsistencies():
    """Test input size inconsistencies."""
    with tf.Session() as sess:
        with pytest.raises(tf.errors.InvalidArgumentError):
            sess.run(tfphasespace.generate([B_AT_REST, BS_AT_REST],
                                           [PION_MASS, PION_MASS, PION_MASS],
                                           n_events=5))
        with pytest.raises(tf.errors.InvalidArgumentError):
            sess.run(tfphasespace.generate(B_AT_REST,
                                           [[PION_MASS, PION_MASS, PION_MASS],
                                            [PION_MASS, PION_MASS, PION_MASS]],
                                           n_events=5))
        with pytest.raises(tf.errors.InvalidArgumentError):
            sess.run(tfphasespace.generate([B_AT_REST, BS_AT_REST],
                                           [[PION_MASS, PION_MASS, PION_MASS],
                                            [PION_MASS, PION_MASS, PION_MASS],
                                            [PION_MASS, PION_MASS, PION_MASS]],
                                           n_events=5))
        with pytest.raises(tf.errors.InvalidArgumentError):
            sess.run(tfphasespace.generate(B_AT_REST,
                                           [6000.0, PION_MASS, PION_MASS]))

# EOF
