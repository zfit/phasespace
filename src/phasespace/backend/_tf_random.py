__all__ = [
    "Generator",
    "from_seed",
    "default_rng",
]

from tensorflow.random import Generator
from tensorflow.random import get_global_generator as default_rng

from_seed = Generator.from_seed
