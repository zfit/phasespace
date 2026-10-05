"""Top-level package for TensorFlow PhaseSpace."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("phasespace")
except PackageNotFoundError:
    pass

__author__ = """Albert Puig Navarro"""
__email__ = "apuignav@gmail.com"
__maintainer__ = "zfit"

__credits__ = ["Jonas Eschle <Jonas.Eschle@cern.ch>"]

__all__ = ["GenParticle", "nbody_decay", "numpy", "random", "to_vectors"]

from tensorflow.experimental import numpy

from . import random
from .phasespace import GenParticle, nbody_decay, to_vectors


def _set_eager_mode():
    import os

    import tensorflow as tf

    is_eager = bool(os.environ.get("PHASESPACE_EAGER"))
    tf.config.run_functions_eagerly(is_eager)


_set_eager_mode()
