from __future__ import annotations

"""This submodule makes it possible for `phasespace` and `DecayLanguage` to work together.

More generally, the `GenMultiDecay` object can also be used as a high-level interface for simulating particles
that can decay in multiple different ways.
"""
import sys

from .genmultidecay import GenMultiDecay  # noqa: F401

try:
    import zfit  # noqa: F401
    import zfit_physics as zphys  # noqa: F401
    from particle import Particle  # noqa: F401
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "The fromdecay functionality in phasespace requires particle and zfit-physics. "
        "Either install phasespace[fromdecay] or particle and zfit-physics.",
        file=sys.stderr,
    ) from error


__all__ = ("GenMultiDecay",)


def __dir__() -> tuple[str, ...]:
    return __all__
