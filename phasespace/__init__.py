# -*- coding: utf-8 -*-

"""Top-level package for Tensorflow PhaseSpace."""
import warnings

__author__ = """Albert Puig Navarro"""
__email__ = 'albert.puig@cern.ch'
__version__ = '0.1.0'

import sys

if sys.version_info < (3, 6):
    warnings.warn("Python 3.5 is NOT supported officially and may be already broken. "
                  "Please consider upgrading to a newer Python version.")

from phasespace.phasespace import generate, Particle
