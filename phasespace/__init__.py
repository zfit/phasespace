"""Top-level package for TensorFlow PhaseSpace."""
import sys

if sys.version_info < (3, 8):
    from importlib_metadata import PackageNotFoundError, version
else:
    from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("phasespace")
except PackageNotFoundError:
    pass

__author__ = """Albert Puig Navarro"""
__email__ = "apuignav@gmail.com"
__maintainer__ = "zfit"

__credits__ = ["Jonas Eschle <Jonas.Eschle@cern.ch>"]

__all__ = ["nbody_decay", "GenParticle", "random"]

import tensorflow as tf

from . import random
from .phasespace import GenParticle, nbody_decay


def _set_eager_mode():
    import os

    is_eager = bool(os.environ.get("PHASESPACE_EAGER"))
    tf.config.run_functions_eagerly(is_eager)


_set_eager_mode()
