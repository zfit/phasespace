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

from tfphasespace import Particle

import decays


def test_name_clashes():
    """Test clashes in particle naming."""
    # In children
    top_part = Particle('B')
    with pytest.raises(KeyError):
        top_part.set_children(['Kstarz', 'Kstarz'], lambda momentum: (decays.KSTARZ_MASS, decays.KSTARZ_MASS))
    top_part = Particle('Top')
    with pytest.raises(KeyError):
        top_part.set_children(['Top', 'Kstarz'], lambda momentum: (decays.KSTARZ_MASS, decays.KSTARZ_MASS))
    # In grandchildren
    top_part = Particle('B')
    kst1, kst2 = top_part.set_children(['Kstarz0', 'Kstarz1'], lambda momentum: (decays.KSTARZ_MASS, decays.KSTARZ_MASS))
    kst1.set_children(['K+', 'pi-'], lambda momentum: (decays.KAON_MASS, decays.PION_MASS))
    with pytest.raises(KeyError):
        kst2.set_children(['K+', 'pi-_1'], lambda momentum: (decays.KAON_MASS, decays.PION_MASS))


def test_kstargamma():
    """Test B0 -> K*gamma."""
    with tf.Session() as sess:
        norm_weights, particles = sess.run(decays.b0_to_kstar_gamma(1000).generate(decays.B0_AT_REST, 1000))
    assert len(norm_weights) == 1000
    assert all([weight < 1 for weight in norm_weights])
    assert len(particles) == 4
    assert set(particles.keys()) == {'K*0', 'gamma', 'K+', 'pi-'}
    assert all([part.shape == (4, 1000) for part in particles.values()])


def test_k1gamma():
    """Test B+ -> K1 (K*pi) gamma."""
    with tf.Session() as sess:
        norm_weights, particles = sess.run(decays.bp_to_k1_kstar_pi_gamma(1000).generate(decays.B0_AT_REST, 1000))
    assert len(norm_weights) == 1000
    assert all([weight < 1 for weight in norm_weights])
    assert len(particles) == 6
    assert set(particles.keys()) == {'K1+', 'K*0', 'gamma', 'K+', 'pi-', 'pi+'}
    assert all([part.shape == (4, 1000) for part in particles.values()])


if __name__ == "__main__":
    test_kstargamma()
    test_k1gamma()

# EOF
