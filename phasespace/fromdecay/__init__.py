import sys
from typing import Tuple

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


__all__ = ("GenMultiDecay", "MASS_WIDTH_TOLERANCE", "DEFAULT_MASS_FUNC")


def __dir__() -> Tuple[str, ...]:
    return __all__
