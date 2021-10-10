import sys

from .fulldecay import FullDecay  # noqa: F401

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
