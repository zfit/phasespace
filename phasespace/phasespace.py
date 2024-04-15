#!/usr/bin/env python3
# =============================================================================
# @file   phasespace.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   25.02.2019
# =============================================================================
"""Implementation of the Raubold and Lynch method to generate n-body events.

The code is based on the GENBOD function (W515 from CERNLIB), documented in:

F. James, Monte Carlo Phase Space, CERN 68-15 (1968)
"""

from __future__ import annotations

import tensorflow.experimental.numpy as _np

from .generation.genparticle import (
    GenParticle,
    Particle,
    generate_decay,
    nbody_decay,
    to_vectors,
)

numpy = _np  # yes, this is weird, but static linters don't like it otherwise with "as"

__all__ = [
    "GenParticle",
    "Particle",
    "generate_decay",
    "nbody_decay",
    "to_vectors",
    "numpy",
]
