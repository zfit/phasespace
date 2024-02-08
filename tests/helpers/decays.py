#!/usr/bin/env python3
# =============================================================================
# @file   decays.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   07.03.2019
# =============================================================================
"""Some physics models to test with."""

try:
    import tensorflow_probability as tfp
except ImportError:
    pass

from phasespace import GenParticle
from phasespace.backend import tnp

# Use RapidSim values (https://github.com/gcowan/RapidSim/blob/master/config/particles.dat)
B0_MASS = 5279.58
PION_MASS = 139.57018
KAON_MASS = 493.677
K1_MASS = 1272.0
K1_WIDTH = 90.0
KSTARZ_MASS = 895.81
KSTARZ_WIDTH = 47.4


def b0_to_kstar_gamma(kstar_width=KSTARZ_WIDTH):
    """Generate B0 -> K*gamma."""

    def kstar_mass(min_mass, max_mass, n_events):
        min_mass = tnp.asarray(min_mass, tnp.float64)
        max_mass = tnp.asarray(max_mass, tnp.float64)
        kstar_width_cast = tnp.asarray(kstar_width, tnp.float64)
        kstar_mass_cast = tnp.asarray(KSTARZ_MASS, dtype=tnp.float64)

        kstar_mass = tnp.broadcast_to(kstar_mass_cast, shape=(n_events,))
        if kstar_width > 0:
            kstar_mass = tfp.distributions.TruncatedNormal(
                loc=kstar_mass, scale=kstar_width_cast, low=min_mass, high=max_mass
            ).sample()
        return kstar_mass

    return GenParticle("B0", B0_MASS).set_children(
        GenParticle("K*0", mass=kstar_mass).set_children(
            GenParticle("K+", mass=KAON_MASS), GenParticle("pi-", mass=PION_MASS)
        ),
        GenParticle("gamma", mass=0.0),
    )


def bp_to_k1_kstar_pi_gamma(k1_width=K1_WIDTH, kstar_width=KSTARZ_WIDTH):
    """Generate B+ -> K1 (-> K* (->K pi) pi) gamma."""

    def res_mass(mass, width, min_mass, max_mass, n_events):
        mass = tnp.asarray(mass, tnp.float64)
        width = tnp.asarray(width, tnp.float64)
        min_mass = tnp.asarray(min_mass, tnp.float64)
        max_mass = tnp.asarray(max_mass, tnp.float64)
        masses = tnp.broadcast_to(mass, shape=(n_events,))
        if kstar_width > 0:
            masses = tfp.distributions.TruncatedNormal(
                loc=masses, scale=width, low=min_mass, high=max_mass
            ).sample()
        return masses

    def k1_mass(min_mass, max_mass, n_events):
        return res_mass(K1_MASS, k1_width, min_mass, max_mass, n_events)

    def kstar_mass(min_mass, max_mass, n_events):
        return res_mass(KSTARZ_MASS, kstar_width, min_mass, max_mass, n_events)

    return GenParticle("B+", B0_MASS).set_children(
        GenParticle("K1+", mass=k1_mass).set_children(
            GenParticle("K*0", mass=kstar_mass).set_children(
                GenParticle("K+", mass=KAON_MASS), GenParticle("pi-", mass=PION_MASS)
            ),
            GenParticle("pi+", mass=PION_MASS),
        ),
        GenParticle("gamma", mass=0.0),
    )


# EOF
