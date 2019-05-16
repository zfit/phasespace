#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   decays.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   07.03.2019
# =============================================================================
"""Some physics models to test with."""

import tensorflow as tf

import tensorflow_probability as tfp

from phasespace import Particle

# Use RapidSim values (https://github.com/gcowan/RapidSim/blob/master/config/particles.dat)
B0_MASS = 5279.58
PION_MASS = 139.57018
KAON_MASS = 493.677
K1_MASS = 1272
K1_WIDTH = 90
KSTARZ_MASS = 895.81
KSTARZ_WIDTH = 47.4


def b0_to_kstar_gamma(kstar_width=KSTARZ_WIDTH):
    """Generate B0 -> K*gamma."""
    def kstar_mass(min_mass, max_mass, n_events):
        ones = tf.ones((n_events, ), dtype=tf.float64)
        kstar_mass = KSTARZ_MASS * ones
        if kstar_width > 0:
            kstar_mass = tfp.distributions.TruncatedNormal(loc=kstar_mass,
                                                           scale=ones * kstar_width,
                                                           low=min_mass,
                                                           high=max_mass).sample()
        return kstar_mass

    return Particle('B0', B0_MASS).set_children(Particle('K*0', mass=kstar_mass)
                                                .set_children(Particle('K+', mass=KAON_MASS),
                                                              Particle('pi-', mass=PION_MASS)),
                                                Particle('gamma', mass=0.0))


def bp_to_k1_kstar_pi_gamma(k1_width=K1_WIDTH, kstar_width=KSTARZ_WIDTH):
    """Generate B+ -> K1 (-> K* (->K pi) pi) gamma."""
    def res_mass(mass, width, min_mass, max_mass, n_events):
        ones = tf.ones((n_events,), dtype=tf.float64)
        masses = mass * ones
        if width > 0:
            min_mass = tf.reshape(min_mass, (n_events,))
            max_mass = tf.reshape(max_mass, (n_events,))
            masses = tfp.distributions.TruncatedNormal(loc=masses,
                                                       scale=ones * width,
                                                       low=min_mass,
                                                       high=max_mass).sample()
        return masses

    def k1_mass(min_mass, max_mass, n_events):
        return res_mass(K1_MASS, k1_width, min_mass, max_mass, n_events)

    def kstar_mass(min_mass, max_mass, n_events):
        return res_mass(KSTARZ_MASS, kstar_width, min_mass, max_mass, n_events)

    return Particle('B+', B0_MASS).set_children(Particle('K1+', mass=k1_mass)
                                                .set_children(Particle('K*0', mass=kstar_mass)
                                                              .set_children(Particle('K+', mass=KAON_MASS),
                                                                            Particle('pi-', mass=PION_MASS)),
                                                              Particle('pi+', mass=PION_MASS)),
                                                Particle('gamma', mass=0.0))

# EOF
