"""Phase space generation with TensorFlow, NumPy, and JAX."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("phasespace")
except PackageNotFoundError:
    pass

__all__ = [
    "GenParticle",
    "get_backend",
    "nbody_decay",
    "random",
    "set_backend",
    "to_vectors",
]

from . import random
from .backend import get_backend, set_backend
from .phasespace import GenParticle, nbody_decay, to_vectors
