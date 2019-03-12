#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   test_chain.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   01.03.2019
# =============================================================================
"""Test decay chain tools."""

from __future__ import print_function, division, absolute_import

import pytest

import tensorflow as tf

from phasespace import Particle

import os
import sys

sys.path.append(os.path.dirname(__file__))

from .helpers import decays


def test_name_clashes():
    """Test clashes in particle naming."""
    # In children
    with pytest.raises(KeyError):
        Particle('Top').set_children(Particle('Kstarz', mass=decays.KSTARZ_MASS),
                                     Particle('Kstarz', mass=decays.KSTARZ_MASS))
    # With itself
    with pytest.raises(KeyError):
        Particle('Top').set_children(Particle('Top', mass=decays.KSTARZ_MASS),
                                     Particle('Kstarz', mass=decays.KSTARZ_MASS))
    # In grandchildren
    with pytest.raises(KeyError):
        Particle('Top').set_children(Particle('Kstarz0', mass=decays.KSTARZ_MASS)
                                     .set_children(Particle('K+', mass=decays.KAON_MASS),
                                                   Particle('pi-', mass=decays.PION_MASS)),
                                     Particle('Kstarz0', mass=decays.KSTARZ_MASS)
                                     .set_children(Particle('K+', mass=decays.KAON_MASS),
                                                   Particle('pi-_1', mass=decays.PION_MASS)))


def test_kstargamma():
    """Test B0 -> K*gamma."""
    with tf.Session() as sess:
        norm_weights, particles = sess.run(decays.b0_to_kstar_gamma().generate(decays.B0_AT_REST, 1000))
    assert len(norm_weights) == 1000
    assert all([weight < 1 for weight in norm_weights])
    assert len(particles) == 4
    assert set(particles.keys()) == {'K*0', 'gamma', 'K+', 'pi-'}
    assert all([part.shape == (4, 1000) for part in particles.values()])


def test_k1gamma():
    """Test B+ -> K1 (K*pi) gamma."""
    with tf.Session() as sess:
        norm_weights, particles = sess.run(decays.bp_to_k1_kstar_pi_gamma().generate(decays.B0_AT_REST, 1000))
    assert len(norm_weights) == 1000
    assert all([weight < 1 for weight in norm_weights])
    assert len(particles) == 6
    assert set(particles.keys()) == {'K1+', 'K*0', 'gamma', 'K+', 'pi-', 'pi+'}
    assert all([part.shape == (4, 1000) for part in particles.values()])


if __name__ == "__main__":
    test_name_clashes()
    test_kstargamma()
    test_k1gamma()

# EOF
