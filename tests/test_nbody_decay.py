#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   test_nbody_decay.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   14.06.2019
# =============================================================================
"""Test n-body decay generator."""

import pytest

from phasespace import nbody_decay
from .helpers import decays

B0_MASS = decays.B0_MASS
PION_MASS = decays.PION_MASS


def test_no_names():
    """Test particle naming when no name is given."""
    decay = nbody_decay(B0_MASS, [PION_MASS, PION_MASS, PION_MASS])
    assert decay.name == 'top'
    assert all(part.name == f"p_{part_num}"
               for part_num, part in enumerate(decay.children))


def test_top_name():
    """Test particle naming when only top name is given."""
    decay = nbody_decay(B0_MASS, [PION_MASS, PION_MASS, PION_MASS],
                        top_name="B0")
    assert decay.name == 'B0'
    assert all(part.name == f"p_{part_num}"
               for part_num, part in enumerate(decay.children))


def test_children_names():
    """Test particle naming when only children names are given."""
    children_names = [f"pion_{i}" for i in range(3)]
    decay = nbody_decay(B0_MASS, [PION_MASS, PION_MASS, PION_MASS],
                        names=children_names)
    assert decay.name == 'top'
    assert children_names == [part.name for part in decay.children]


def test_all_names():
    """Test particle naming when all names are given."""
    children_names = [f"pion_{i}" for i in range(3)]
    decay = nbody_decay(B0_MASS, [PION_MASS, PION_MASS, PION_MASS],
                        top_name="B0", names=children_names)
    assert decay.name == 'B0'
    assert children_names == [part.name for part in decay.children]


def test_mismatching_names():
    """Test wrong number of names given for children."""
    children_names = [f"pion_{i}" for i in range(4)]
    with pytest.raises(ValueError):
        nbody_decay(B0_MASS, [PION_MASS, PION_MASS, PION_MASS],
                    names=children_names)

# EOF
