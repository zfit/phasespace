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

from phasespace import GenParticle

import os
import sys

sys.path.append(os.path.dirname(__file__))

from .helpers import decays

def setup_method():
    GenParticle._sess.close()
    tf.reset_default_graph()



def test_name_clashes():
    """Test clashes in particle naming."""
    # In children
    with pytest.raises(KeyError):
        GenParticle('Top', 0).set_children(GenParticle('Kstarz', mass=decays.KSTARZ_MASS),
                                           GenParticle('Kstarz', mass=decays.KSTARZ_MASS))
    # With itself
    with pytest.raises(KeyError):
        GenParticle('Top', 0).set_children(GenParticle('Top', mass=decays.KSTARZ_MASS),
                                           GenParticle('Kstarz', mass=decays.KSTARZ_MASS))
    # In grandchildren
    with pytest.raises(KeyError):
        GenParticle('Top', 0).set_children(GenParticle('Kstarz0', mass=decays.KSTARZ_MASS)
                                           .set_children(GenParticle('K+', mass=decays.KAON_MASS),
                                                         GenParticle('pi-', mass=decays.PION_MASS)),
                                           GenParticle('Kstarz0', mass=decays.KSTARZ_MASS)
                                           .set_children(GenParticle('K+', mass=decays.KAON_MASS),
                                                         GenParticle('pi-_1', mass=decays.PION_MASS)))

def test_wrong_children():
    """Test wrong number of children."""
    with pytest.raises(ValueError):
        GenParticle('Top', 0).set_children(GenParticle('Kstarz0', mass=decays.KSTARZ_MASS))


def test_grandchildren():
    """Test that grandchildren detection is correct."""
    top = GenParticle('Top', 0)
    assert not top.has_children
    assert not top.has_grandchildren
    assert not top.set_children(GenParticle('Child1', mass=decays.KSTARZ_MASS),
                                GenParticle('Child2', mass=decays.KSTARZ_MASS)).has_grandchildren


def test_reset_children():
    """Test when children are set twice."""
    top = GenParticle('Top', 0).set_children(GenParticle('Child1', mass=decays.KSTARZ_MASS),
                                             GenParticle('Child2', mass=decays.KSTARZ_MASS))
    with pytest.raises(ValueError):
        top.set_children(GenParticle('Child3', mass=decays.KSTARZ_MASS),
                         GenParticle('Child4', mass=decays.KSTARZ_MASS))


def test_no_children():
    """Test when no children have been configured."""
    top = GenParticle('Top', 0)
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


def test_repr():
    """Test string representation."""
    b0 = decays.b0_to_kstar_gamma()
    kst = b0.children[0]
    assert str(b0) == "<phasespace.GenParticle: name='B0' mass=5279.58 children=[K*0, gamma]>"
    assert str(kst) == "<phasespace.GenParticle: name='K*0' mass=variable children=[K+, pi-]>"


if __name__ == "__main__":
    test_name_clashes()
    test_kstargamma()
    test_k1gamma()

# EOF
