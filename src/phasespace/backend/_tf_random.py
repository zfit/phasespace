__all__ = [
    "Generator",
    "default_rng",
]

from tensorflow.random import Generator
from tensorflow.random import get_global_generator as default_rng
