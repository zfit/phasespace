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

from tfphasespace import Particle
from tfphasespace.kinematics import mass

# Use RapidSim values (https://github.com/gcowan/RapidSim/blob/master/config/particles.dat)
B0_MASS = 5279.58
B0_AT_REST = tf.stack((0.0, 0.0, 0.0, B0_MASS), axis=-1)
PION_MASS = 139.57018
KAON_MASS = 493.677
K1_MASS = 1272
K1_WIDTH = 90
KSTARZ_MASS = 895.81
KSTARZ_WIDTH = 47.4


def b0_to_kstar_gamma(n_events, kstar_width=KSTARZ_WIDTH):
    """Generate B0 -> K*gamma."""
    def kstar_gamma_masses(momentum):
        if len(momentum.shape) > 1:
            shape = (momentum.shape[1], )
        else:
            shape = (n_events, )
        if kstar_width == 0:
            kstar_mass = KSTARZ_MASS * tf.ones(shape)
        else:
            kstar_mass = tf.random.normal(shape, KSTARZ_MASS, kstar_width)
        return kstar_mass, tf.zeros(shape)

    top_part = Particle('B0')
    kstar, _ = top_part.set_children(['K*0', 'gamma'], kstar_gamma_masses)
    kstar.set_children(['K+', 'pi-'], lambda momentum: (KAON_MASS, PION_MASS))
    return top_part


def bp_to_k1_kstar_pi_gamma(n_events, k1_width=K1_WIDTH, kstar_width=KSTARZ_WIDTH):
    """Generate B+ -> K1 (-> K* (->K pi) pi) gamma."""
    def get_k1_gamma_masses(momentum):
        if len(momentum.shape) > 1:
            shape = momentum.shape[1]
        else:
            shape = (n_events, )
        if k1_width == 0:
            k1_mass = K1_MASS * tf.ones(shape)
        else:
            k1_mass = tf.random.normal(shape, K1_MASS, k1_width)
        return k1_mass, tf.zeros(shape)

    def get_kstarpi_masses(momentum):
        top_mass = mass(momentum)
        shape = top_mass.shape.as_list()
        ones = tf.ones(shape, dtype=tf.float64)
        if kstar_width == 0:
            kst_masses = ones * KSTARZ_MASS
        else:
            kst_masses = tfp.distributions.TruncatedNormal(loc=ones * KSTARZ_MASS,
                                                           scale=ones * kstar_width,
                                                           low=ones * (KAON_MASS + PION_MASS),
                                                           high=top_mass - PION_MASS).sample()
        return kst_masses, ones * PION_MASS

    top_part = Particle('B+')
    k1, _ = top_part.set_children(['K1+', 'gamma'], get_k1_gamma_masses)
    kstar, _ = k1.set_children(['K*0', 'pi+'], get_kstarpi_masses)
    kstar.set_children(['K+', 'pi-'], lambda momentum: (KAON_MASS, PION_MASS))
    return top_part

# EOF
