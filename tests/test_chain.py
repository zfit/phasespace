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


B_MASS = 5279.0
B_AT_REST = tf.stack((0.0, 0.0, 0.0, B_MASS), axis=-1)
BS_AT_REST = tf.stack((0.0, 0.0, 0.0, B_MASS + 86.8), axis=-1)
PION_MASS = 139.6
KAON_MASS = 493.7
KSTARZ_MASS = 891.8


def test_name_clashes():
    """Test clashes in particle naming."""
    # In children
    top_part = Particle('B')
    with pytest.raises(KeyError):
        top_part.set_children(['Kstarz', 'Kstarz'], lambda momentum: (KSTARZ_MASS, KSTARZ_MASS))
    top_part = Particle('Top')
    with pytest.raises(KeyError):
        top_part.set_children(['Top', 'Kstarz'], lambda momentum: (KSTARZ_MASS, KSTARZ_MASS))
    # In grandchildren
    top_part = Particle('B')
    kst1, kst2 = top_part.set_children(['Kstarz0', 'Kstarz1'], lambda momentum: (KSTARZ_MASS, KSTARZ_MASS))
    kst1.set_children(['K+', 'pi-'], lambda momentum: (KAON_MASS, PION_MASS))
    with pytest.raises(KeyError):
        kst2.set_children(['K+', 'pi-_1'], lambda momentum: (KAON_MASS, PION_MASS))


# EOF
