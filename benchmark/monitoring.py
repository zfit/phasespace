#!/usr/bin/env python
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

    def __init__(self, verbose=False):
        """Initialize the timer."""
        self.verbose = verbose
        self._timer = default_timer
        self.start = 0
        self.elapsed = 0

    def __enter__(self):
        self.start = self._timer()
        return self

    def __exit__(self, *args):
        self.elapsed = self._timer() - self.start
        if self.verbose:
            print('Elapsed time: {} ms'.format(self.elapsed * 1000.0))

# EOF
