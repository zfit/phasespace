from .fulldecay import FullDecay

import sys
try:
    from particle import Particle
    import zfit
    import zfit_physics as zphys
except ModuleNotFoundError:
    print(
        "the fulldecay functionality in phasespace requires particle and zfit-physics. "
        "Either install phasespace[fulldecay] or particle and zfit-physics.",
        file=sys.stderr,
    )
    raise
