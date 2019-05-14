#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   test_generate.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   27.02.2019
# =============================================================================
"""Basic dimensionality tests."""

import pytest

import tensorflow as tf

import phasespace

import os
import sys

sys.path.append(os.path.dirname(__file__))

from .helpers import decays

B0_MASS = decays.B0_MASS
PION_MASS = decays.PION_MASS


def test_one_event():
    """Test B->pi pi pi."""
    norm_weights, particles = phasespace.generate_decay(B0_MASS,
                                                        [PION_MASS, PION_MASS, PION_MASS])
    assert len(norm_weights) == 1
    assert all([weight < 1 for weight in norm_weights])
    assert len(particles) == 3
    assert all([part.shape == (4, 1) for part in particles])


def test_one_event_tf():
    """Test B->pi pi pi."""
    with tf.Session() as sess:
        norm_weights, particles = sess.run(phasespace.generate_decay(B0_MASS,
                                                                     [PION_MASS, PION_MASS, PION_MASS]))
        assert len(norm_weights) == 1
        assert all([weight < 1 for weight in norm_weights])
        assert len(particles) == 3
        assert all([part.shape == (4, 1) for part in particles])


@pytest.mark.parametrize("n_events", argvalues=[5, tf.constant(5), tf.Variable(initial_value=5)])
def test_n_events(n_events):
    """Test 5 B->pi pi pi."""
    norm_weights, particles = phasespace.generate_decay(B0_MASS,
                                                        [PION_MASS, PION_MASS, PION_MASS],
                                                        n_events=n_events)
    assert len(norm_weights) == 5
    assert all([weight < 1 for weight in norm_weights])
    assert len(particles) == 3
    assert all([part.shape == (4, 5) for part in particles])


if __name__ == "__main__":
    test_one_event()
    test_n_events(5)

# EOF
