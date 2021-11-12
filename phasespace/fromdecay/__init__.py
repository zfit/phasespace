"""This submodule makes it possible for `phasespace` and.

[`DecayLanguage`](https://github.com/scikit-hep/decaylanguage/) to work together.
More generally, the `GenMultiDecay` object can also be used as a high-level interface for simulating particles
that can decay in multiple different ways.
"""
import sys
from typing import Tuple

from .genmultidecay import (  # noqa: F401
    DEFAULT_MASS_FUNC,
    MASS_WIDTH_TOLERANCE,
    GenMultiDecay,
)

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


__all__ = ("GenMultiDecay", "MASS_WIDTH_TOLERANCE", "DEFAULT_MASS_FUNC")


def __dir__() -> Tuple[str, ...]:
    return __all__
