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

import pytest

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


if __name__ == "__main__":
    test_one_event()
    test_n_events(5)

    # EOF
