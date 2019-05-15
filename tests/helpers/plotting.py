#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   plotting.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   07.03.2019
# =============================================================================
"""Plotting helpers for tests."""

import numpy as np


def make_norm_histo(array, range_, weights=None):
    """Make histo and modify dimensions."""
    histo = np.histogram(array, 100, range=range_, weights=weights)[0]
    return histo / np.sum(histo)


def mass(vector):
    """Calculate mass scalar for Lorentz 4-momentum."""
    return np.sqrt(np.sum(vector * vector * np.reshape(np.array([-1., -1., -1., 1.]),
                                                       (1, 4)),
                          axis=1))

# EOF
