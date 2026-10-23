"""Random number generation for the selected computational backend."""

from __future__ import annotations

from typing import Protocol

from . import backend


class RandomGenerator(Protocol):
    def uniform(self, *args, **kwargs): ...


SeedLike = int | RandomGenerator | None


def get_rng(seed: SeedLike = None) -> RandomGenerator:
    """Return a generator for the current backend.

    An integer starts a reproducible stream. ``None`` uses the backend's default
    generator. Passing a generator reuses and advances that stream. Generators
    belong to one backend and cannot be reused after switching to another.
    """
    if seed is None:
        return backend.random.default_rng()
    if isinstance(seed, backend.random.Generator):
        return seed
    if hasattr(seed, "uniform"):
        raise ValueError("The random generator does not belong to the current backend")
    return backend.random.from_seed(seed)


def generate_uniform(
    rng: RandomGenerator, shape: tuple[int, ...], minval=0, maxval=1, dtype=None
):
    if dtype is None:
        dtype = backend.tnp.float64
    return backend.random.uniform(rng, shape, minval, maxval, dtype)
