# -*- coding: utf-8 -*-

"""Top-level package for TensorFlow PhaseSpace."""

__author__ = """Albert Puig Navarro"""
__email__ = 'albert.puig@cern.ch'
__version__ = '1.0.0'
__maintainer__ = "zfit"

__credits__ = ["Jonas Eschle <jonas.eschle@cern.ch>"]

__all__ = ['generate_decay', 'Particle']

from .phasespace import generate_decay, Particle
