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


B_MASS = 5279.0
B_AT_REST = tf.stack((0.0, 0.0, 0.0, B_MASS), axis=-1)
BS_AT_REST = tf.stack((0.0, 0.0, 0.0, B_MASS + 86.8), axis=-1)
PION_MASS = 139.6


def test_one_event():
    """Test B->pi pi pi."""
    norm_weights, particles = tf.Session().run(tfphasespace.generate(B_AT_REST,
                                                                     [PION_MASS, PION_MASS, PION_MASS]))
    assert len(norm_weights) == 1
    assert all([weight < 1 for weight in norm_weights])
    assert len(particles) == 3
    assert all([part.shape == (4, 1) for part in particles])


def test_n_events():
    """Test 5 B->pi pi pi."""
    norm_weights, particles = tf.Session().run(tfphasespace.generate(B_AT_REST,
                                                                     [PION_MASS, PION_MASS, PION_MASS],
                                                                     n_events=5))
    assert len(norm_weights) == 5
    assert all([weight < 1 for weight in norm_weights])
    assert len(particles) == 3
    assert all([part.shape == (4, 5) for part in particles])


def test_n_events_implicit_parent():
    """Test multiple events by passing mutiple parent momenta."""
    norm_weights, particles = tf.Session().run(tfphasespace.generate([B_AT_REST, BS_AT_REST],
                                                                     [PION_MASS, PION_MASS, PION_MASS]))
    assert len(norm_weights) == 2
    assert all([weight < 1 for weight in norm_weights])
    assert len(particles) == 3
    assert all([part.shape == (4, 2) for part in particles])


def test_n_events_implicit_daughters():
    """Test multiple events by passing mutiple daugther masses."""
    norm_weights, particles = tf.Session().run(tfphasespace.generate(B_AT_REST,
                                                                     [[PION_MASS, PION_MASS, PION_MASS],
                                                                      [PION_MASS, PION_MASS, PION_MASS]]))
    assert len(norm_weights) == 2
    assert all([weight < 1 for weight in norm_weights])
    assert len(particles) == 3
    assert all([part.shape == (4, 2) for part in particles])


def test_input_inconsistencies():
    """Test input size inconsistencies."""
    with pytest.raises(ValueError):
        tf.Session().run(tfphasespace.generate([B_AT_REST, BS_AT_REST],
                                               [PION_MASS, PION_MASS, PION_MASS],
                                               n_events=5))
    with pytest.raises(ValueError):
        tf.Session().run(tfphasespace.generate(B_AT_REST,
                                               [[PION_MASS, PION_MASS, PION_MASS],
                                                [PION_MASS, PION_MASS, PION_MASS]],
                                               n_events=5))
    with pytest.raises(ValueError):
        tf.Session().run(tfphasespace.generate([B_AT_REST, BS_AT_REST],
                                               [[PION_MASS, PION_MASS, PION_MASS],
                                                [PION_MASS, PION_MASS, PION_MASS],
                                                [PION_MASS, PION_MASS, PION_MASS]],
                                               n_events=5))
    with pytest.raises(tf.errors.InvalidArgumentError):
        tf.Session().run(tfphasespace.generate(B_AT_REST,
                                               [B_MASS, PION_MASS, PION_MASS]))

# EOF
