#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   test_generate.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   27.02.2019
# =============================================================================
"""Basic dimensionality tests."""

import os
import sys

import numpy as np
import pytest
import tensorflow as tf

import phasespace

sys.path.append(os.path.dirname(__file__))

from .helpers import decays

B0_MASS = decays.B0_MASS
PION_MASS = decays.PION_MASS


def test_one_event():
    """Test B->pi pi pi."""
    decay = phasespace.nbody_decay(B0_MASS, [PION_MASS, PION_MASS, PION_MASS])
    norm_weights, particles = decay.generate(n_events=1)
    assert norm_weights.shape[0] == 1
    assert all([weight.numpy() < 1 for weight in norm_weights])
    assert len(particles) == 3
    assert all([part.shape == (1, 4) for part in particles.values()])


def test_one_event_tf():
    """Test B->pi pi pi."""
    decay = phasespace.nbody_decay(B0_MASS, [PION_MASS, PION_MASS, PION_MASS])
    norm_weights, particles = decay.generate(n_events=1)

    assert norm_weights.shape[0] == 1
    assert all([weight.numpy() < 1 for weight in norm_weights])
    assert len(particles) == 3
    assert all([part.shape == (1, 4) for part in particles.values()])


@pytest.mark.parametrize("n_events", argvalues=[5])
def test_n_events(n_events):
    """Test 5 B->pi pi pi."""
    decay = phasespace.nbody_decay(B0_MASS, [PION_MASS, PION_MASS, PION_MASS])
    norm_weights, particles = decay.generate(n_events=n_events)
    assert norm_weights.shape[0] == 5
    assert all([weight.numpy() < 1 for weight in norm_weights])
    assert len(particles) == 3
    assert all([part.shape == (5, 4) for part in particles.values()])


def test_deterministic_events():
    decay = phasespace.nbody_decay(B0_MASS, [PION_MASS, PION_MASS, PION_MASS])
    common_seed = 36
    norm_weights_seeded1, particles_seeded1 = decay.generate(n_events=100, seed=common_seed)
    norm_weights_global, particles_global = decay.generate(n_events=100)
    norm_weights_rnd, particles_rnd = decay.generate(n_events=100, seed=152)
    norm_weights_seeded2, particles_seeded2 = decay.generate(n_events=100, seed=common_seed)

    # # test global deterministic behavior
    # tf.random.set_seed(15)
    # norm_weights_global_seeded1, particles_global_seeded1 = decay.generate(n_events=100)
    # tf.random.set_seed(15)
    # norm_weights_global_seeded2, particles_global_seeded2 = decay.generate(n_events=100)

    np.testing.assert_allclose(norm_weights_seeded1, norm_weights_seeded2)
    for part1, part2 in zip(particles_seeded1.values(), particles_seeded2.values()):
        np.testing.assert_allclose(part1, part2)

    # np.testing.assert_allclose(norm_weights_global_seeded1, norm_weights_global_seeded2)
    # for part1, part2 in zip(particles_global_seeded1.values(), particles_global_seeded2.values()):
    #     np.testing.assert_allclose(part1, part2)
    #
    # assert not np.allclose(norm_weights_seeded1, norm_weights_global_seeded1)
    # for part1, part2 in zip(particles_seeded1.values(), particles_global_seeded1.values()):
    #     assert not np.allclose(part1, part2)
    #
    # assert not np.allclose(norm_weights_global, norm_weights_global_seeded1)
    # for part1, part2 in zip(particles_global.values(), particles_global_seeded1.values()):
    #     assert not np.allclose(part1, part2)

    assert not np.allclose(norm_weights_seeded1, norm_weights_rnd)
    for part1, part2 in zip(particles_seeded1.values(), particles_rnd.values()):
        assert not np.allclose(part1, part2)

    assert not np.allclose(norm_weights_global, norm_weights_rnd)
    for part1, part2 in zip(particles_global.values(), particles_rnd.values()):
        assert not np.allclose(part1, part2)


if __name__ == "__main__":
    test_one_event()
    test_n_events(5)

    # EOF
