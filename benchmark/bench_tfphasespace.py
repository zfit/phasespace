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

import os, sys

sys.path.append(os.path.dirname(__file__))

from monitoring import Timer

B_MASS = 5279.0
B_AT_REST = tf.stack((0.0, 0.0, 0.0, B_MASS), axis=-1)
PION_MASS = 139.6

N_EVENTS = 1000000
CHUNK_SIZE = 1000000

sess = tf.Session()


def test_three_body():
    """Test B -> pi pi pi decay."""

    do_run()  # to get rid of initial overhead
    print("starting benchmark")
    with Timer(verbose=True):
        do_run()


def do_run():
    sess.run([tfphasespace.generate(B_AT_REST,
                                    [PION_MASS, PION_MASS, PION_MASS],
                                    CHUNK_SIZE)
              for _ in range(0, N_EVENTS + 1, CHUNK_SIZE)])


if __name__ == "__main__":
    test_three_body()

# EOF
