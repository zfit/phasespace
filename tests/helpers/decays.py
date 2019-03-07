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

B_MASS = 5279.0
B_AT_REST = tf.stack((0.0, 0.0, 0.0, B_MASS), axis=-1)
PION_MASS = 139.6
KAON_MASS = 493.7
K1_MASS = 1270
K1_WIDTH = 90
KSTARZ_MASS = 891.8
KSTARZ_WIDTH = 45


def b0_to_kstar_gamma():
    """Generate B0 -> K*gamma."""
    top_part = Particle('B0')
    kstar, _ = top_part.set_children(['K*0', 'gamma'], lambda momentum: (tf.random.normal((1000,), KSTARZ_MASS, 45),
                                                                         tf.zeros((1000,))))
    kstar.set_children(['K+', 'pi-'], lambda momentum: (KAON_MASS, PION_MASS))
    return top_part


def bp_to_k1_kstar_pi_gamma():
    """Generate B+ -> K1 (-> K* (->K pi) pi) gamma."""
    def get_kstarpi_masses(momentum):
        top_mass = mass(momentum)
        shape = top_mass.shape.as_list()
        ones = tf.ones(shape, dtype=tf.float64)
        kst_masses = tfp.distributions.TruncatedNormal(loc=ones * KSTARZ_MASS,
                                                       scale=ones * KSTARZ_WIDTH,
                                                       low=ones * (KAON_MASS + PION_MASS),
                                                       high=top_mass - PION_MASS).sample()
        return kst_masses, ones * PION_MASS

    top_part = Particle('B+')
    k1, _ = top_part.set_children(['K1', 'gamma'],
                                  lambda momentum: (tf.random.normal((1000,), K1_MASS, K1_WIDTH,
                                                                     dtype=tf.float64),
                                                    tf.zeros((1000,), dtype=tf.float64)))
    kstar, _ = k1.set_children(['K*0', 'pi+'], get_kstarpi_masses)
    kstar.set_children(['K+', 'pi-'], lambda momentum: (KAON_MASS, PION_MASS))
    return top_part

# EOF
