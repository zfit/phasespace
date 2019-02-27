#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   bench_tfphasespace.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   27.02.2019
# =============================================================================
"""Benchmark tfphasespace."""

import tensorflow as tf

from tfphasespace import tfphasespace

B_MASS = 5279.0
B_AT_REST = tf.stack((0.0, 0.0, 0.0, B_MASS), axis=-1)
PION_MASS = 139.6


def test_three_body():
    """Test B -> pi pi pi decay."""
    weights, particles = tf.Session().run(tfphasespace.generate(B_AT_REST,
                                                                [PION_MASS, PION_MASS, PION_MASS],
                                                                100000000))

if __name__ == "__main__":
    test_three_body()


# EOF
