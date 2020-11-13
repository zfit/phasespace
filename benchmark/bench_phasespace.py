#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   bench_phasespace.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   27.02.2019
# =============================================================================
"""Benchmark phasespace."""

import os
import sys

import tensorflow as tf

from phasespace import phasespace

sys.path.append(os.path.dirname(__file__))

# from .monitoring import Timer

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# @file   monitoring.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   14.02.2017
# =============================================================================
"""Various code monitoring utilities."""

import os

from timeit import default_timer


def memory_usage():
    """Get memory usage of current process in MiB.

    Tries to use :mod:`psutil`, if possible, otherwise fallback to calling
    ``ps`` directly.

    Return:
        float: Memory usage of the current process.

    """
    pid = os.getpid()
    try:
        import psutil
        process = psutil.Process(pid)
        mem = process.memory_info()[0] / float(2 ** 20)
    except ImportError:
        import subprocess
        out = subprocess.Popen(['ps', 'v', '-p', str(pid)],
                               stdout=subprocess.PIPE).communicate()[0].split(b'\n')
        vsz_index = out[0].split().index(b'RSS')
        mem = float(out[1].split()[vsz_index]) / 1024
    return mem


# pylint: disable=too-few-public-methods
class Timer(object):
    """Time the code placed inside its context.

    Taken from http://coreygoldberg.blogspot.ch/2012/06/python-timer-class-context-manager-for.html

    Attributes:
        verbose (bool): Print the elapsed time at context exit?
        start (float): Start time in seconds since Epoch Time. Value set
            to 0 if not run.
        elapsed (float): Elapsed seconds in the timer. Value set to
            0 if not run.

    Arguments:
        verbose (bool, optional): Print the elapsed time at
            context exit? Defaults to False.

    """

    def __init__(self, verbose=False, n=1):
        """Initialize the timer."""
        self.verbose = verbose
        self.n = n
        self._timer = default_timer
        self.start = 0
        self.elapsed = 0

    def __enter__(self):
        self.start = self._timer()
        return self

    def __exit__(self, *args):
        self.elapsed = self._timer() - self.start
        if self.verbose:
            print('Elapsed time: {} ms'.format(self.elapsed * 1000.0 / self.n))


# EOF


# to play around with optimization, no big effect though
NUM_PARALLEL_EXEC_UNITS = 1
# config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=1,
#                         allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})

B_MASS = 5279.0
B_AT_REST = tf.stack((0.0, 0.0, 0.0, B_MASS), axis=-1)
PION_MASS = 139.6

N_EVENTS = 1000000
CHUNK_SIZE = int(N_EVENTS)

n_runs = 10


# N_EVENTS_VAR = tf.Variable(initial_value=N_EVENTS)
# CHUNK_SIZE_VAR = tf.Variable(initial_value=CHUNK_SIZE)


def test_three_body():
    """Test B -> pi pi pi decay."""
    with Timer(verbose=True):
        print("Initial run (may takes more time than consequent runs)")
        do_run()  # to get rid of initial overhead
    print("starting benchmark")
    with Timer(verbose=True, n=n_runs):
        for _ in range(n_runs):
            # CHUNK_SIZE_VAR.assign(CHUNK_SIZE + 1)  # +1 to make sure we're not using any trivial caching
            samples = do_run()

    print(f"nevents produced {samples[0][0].shape}")
    print("Shape of one particle momentum", samples[0][1]['p_0'].shape)


decay = phasespace.nbody_decay(B_MASS,
                               [PION_MASS, PION_MASS, PION_MASS],
                               )


# tf.config.run_functions_eagerly(True)
@tf.function(autograph=False)
def do_run():
    return [decay.generate(N_EVENTS)
            for _ in range(0, N_EVENTS, CHUNK_SIZE)
            ]


if __name__ == "__main__":
    test_three_body()

# EOF
