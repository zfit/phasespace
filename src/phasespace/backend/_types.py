"""Containers for the library-specific objects that each backend module provides."""

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class Random:
    """Random number API of one backend, following `tf.random.Generator`."""

    Generator: type
    default_rng: Callable[..., Any]
    from_seed: Callable[..., Any]
    uniform: Callable[..., Any]


@dataclass(frozen=True)
class Implementation:
    """Library-specific objects behind the public names of `phasespace.backend`, one instance per backend."""

    array: ModuleType
    random: Random
    Tensor: type
    Variable: type
    shape: Callable[[Any], Any]
    assert_equal: Callable[..., Any]
    assert_greater_equal: Callable[..., Any]
