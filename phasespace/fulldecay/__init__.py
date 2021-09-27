import sys

from .fulldecay import FullDecay

try:
    import zfit
    import zfit_physics as zphys
    from particle import Particle
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "The fulldecay functionality in phasespace requires particle and zfit-physics. "
        "Either install phasespace[fulldecay] or particle and zfit-physics.",
        file=sys.stderr,
    ) from error
