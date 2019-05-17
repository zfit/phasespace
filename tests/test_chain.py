#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   test_chain.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   01.03.2019
# =============================================================================
"""Test decay chain tools."""

import pytest

import tensorflow as tf

from phasespace import Particle

import os
import sys

sys.path.append(os.path.dirname(__file__))

from .helpers import decays

def setup_method():
    Particle._sess.close()
    tf.reset_default_graph()



def test_name_clashes():
    """Test clashes in particle naming."""
    # In children
    with pytest.raises(KeyError):
        Particle('Top', 0).set_children(Particle('Kstarz', mass=decays.KSTARZ_MASS),
                                        Particle('Kstarz', mass=decays.KSTARZ_MASS))
    # With itself
    with pytest.raises(KeyError):
        Particle('Top', 0).set_children(Particle('Top', mass=decays.KSTARZ_MASS),
                                        Particle('Kstarz', mass=decays.KSTARZ_MASS))
    # In grandchildren
    with pytest.raises(KeyError):
        Particle('Top', 0).set_children(Particle('Kstarz0', mass=decays.KSTARZ_MASS)
                                        .set_children(Particle('K+', mass=decays.KAON_MASS),
                                                      Particle('pi-', mass=decays.PION_MASS)),
                                        Particle('Kstarz0', mass=decays.KSTARZ_MASS)
                                        .set_children(Particle('K+', mass=decays.KAON_MASS),
                                                      Particle('pi-_1', mass=decays.PION_MASS)))

def test_wrong_children():
    """Test wrong number of children."""
    with pytest.raises(ValueError):
        Particle('Top', 0).set_children(Particle('Kstarz0', mass=decays.KSTARZ_MASS))


def test_grandchildren():
    """Test that grandchildren detection is correct."""
    top = Particle('Top', 0)
    assert not top.has_children
    assert not top.has_grandchildren
    assert not top.set_children(Particle('Child1', mass=decays.KSTARZ_MASS),
                                Particle('Child2', mass=decays.KSTARZ_MASS)).has_grandchildren


def test_reset_children():
    """Test when children are set twice."""
    top = Particle('Top', 0).set_children(Particle('Child1', mass=decays.KSTARZ_MASS),
                                          Particle('Child2', mass=decays.KSTARZ_MASS))
    with pytest.raises(ValueError):
        top.set_children(Particle('Child3', mass=decays.KSTARZ_MASS),
                         Particle('Child4', mass=decays.KSTARZ_MASS))


def test_no_children():
    """Test when no children have been configured."""
    top = Particle('Top', 0)
    with pytest.raises(ValueError):
        top.generate(n_events=1)


def test_resonance_top():
    """Test when a resonance is used as the top particle."""
    kstar = decays.b0_to_kstar_gamma().children[0]
    with pytest.raises(ValueError):
        kstar.generate(n_events=1)


def test_kstargamma():
    """Test B0 -> K*gamma."""
    norm_weights, particles = decays.b0_to_kstar_gamma().generate(n_events=1000)
    assert len(norm_weights) == 1000
    assert all([weight < 1 for weight in norm_weights])
    assert len(particles) == 4
    assert set(particles.keys()) == {'K*0', 'gamma', 'K+', 'pi-'}
    assert all([part.shape == (1000, 4) for part in particles.values()])


def test_k1gamma():
    """Test B+ -> K1 (K*pi) gamma."""
    norm_weights, particles = decays.bp_to_k1_kstar_pi_gamma().generate(n_events=1000)
    assert len(norm_weights) == 1000
    assert all([weight < 1 for weight in norm_weights])
    assert len(particles) == 6
    assert set(particles.keys()) == {'K1+', 'K*0', 'gamma', 'K+', 'pi-', 'pi+'}
    assert all([part.shape == (1000, 4) for part in particles.values()])


if __name__ == "__main__":
    test_name_clashes()
    test_kstargamma()
    test_k1gamma()

# EOF
