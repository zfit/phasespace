"""Top-level package for TensorFlow PhaseSpace."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("phasespace")
except PackageNotFoundError:
    pass

__author__ = """Albert Puig Navarro"""
__email__ = "apuignav@gmail.com"
__maintainer__ = "zfit"

__credits__ = [
    "Jonas Eschle <Jonas.Eschle@cern.ch>",
]

__all__ = [
    "GenParticle",
    "nbody_decay",
    "random",
    "to_vectors",
]

# Import backend to set eager mode
from . import backend as _backend  # noqa: F401
from . import random
from .phasespace import GenParticle, nbody_decay, to_vectors
