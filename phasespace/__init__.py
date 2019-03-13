# -*- coding: utf-8 -*-

"""Top-level package for TensorFlow PhaseSpace."""
import warnings

__author__ = """Albert Puig Navarro"""
__email__ = 'albert.puig@cern.ch'
__version__ = '0.9.0'

__all__ = ['generate', 'Particle']

import sys

from .phasespace import generate, Particle

if sys.version_info < (3, 6):
    warnings.warn("Python 3.5 is NOT supported officially and may be already broken. "
                  "Please consider upgrading to a newer Python version.")

