#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   test_physics.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   27.02.2019
# =============================================================================
"""Test physics output."""

import os
import platform

import numpy as np
from scipy.stats import ks_2samp


import uproot

import tensorflow as tf

from tfphasespace import tfphasespace


REF_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'bto3pi.root')

B_MASS = 5279.0
B_AT_REST = tf.stack((0.0, 0.0, 0.0, B_MASS), axis=-1)
PION_MASS = 139.6


def create_ref_histos():
    """Load reference histogram data."""
    def make_histos(vector_list, range_):
        """Make histograms."""
        v_array = np.stack([vector_list.x, vector_list.y, vector_list.z, vector_list.E])
        histos = tuple(np.histogram(v_array[coord], 100, range=range_)[0]
                       for coord in range(4))
        return tuple(histo/np.sum(histo) for histo in histos)

    if not os.path.exists(REF_FILE):
        script = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              '..',
                                              'scripts',
                                              'prepare_test_samples.cxx'))
        os.system('root -qb {}+'.format(script))
    events = uproot.open(REF_FILE)['events']
    pions = events.arrays(['pion_1', 'pion_2', 'pion_3'])
    weights = np.histogram(events.array('weight'), 100, range=(0, 1))[0]
    return sum([make_histos(pions[pion], range_=(-3000, 3000))
                for pion in (b'pion_1', b'pion_2', b'pion_3')], tuple()), \
        weights/np.sum(weights)


def test_three_body():
    """Test B -> pi pi pi decay."""
    def make_histo(array, range_):
        """Make histo and modify dimensions."""
        histo = np.histogram(array, 100, range=range_)[0]
        return histo/np.sum(histo)
        # return np.expand_dims(np.histogram(parts[0], 100, range=(-3000, 3000))[0], axis=0)

    weights, particles = tf.Session().run(tfphasespace.generate(B_AT_REST,
                                                                [PION_MASS, PION_MASS, PION_MASS],
                                                                100000))
    parts = np.concatenate(particles, axis=0)
    histos = [make_histo(parts[coord], range_=(-3000, 3000)) for coord in range(parts.shape[0])]
    weight_histos = make_histo(weights, range_=(0, 1))
    ref_histos, ref_weights = create_ref_histos()
    p_values = np.array([ks_2samp(histos[coord], ref_histos[coord])[1]
                         for coord, _ in enumerate(histos)] +
                        [ks_2samp(weight_histos, ref_weights)[1]])
    # Let's plot
    if platform.system() == 'Darwin':
        import matplotlib
        matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    x = np.linspace(-3000, 3000, 100)
    for coord, _ in enumerate(histos):
        plt.hist(x, weights=histos[coord], alpha=0.5, label='tfphasespace', bins=100)
        plt.hist(x, weights=ref_histos[coord], alpha=0.5, label='TGenPhasespace', bins=100)
        plt.legend(loc='upper right')
        plt.savefig("pion_{}_{}.png".format(int(coord / 4) + 1,
                                            ['x', 'y', 'z', 'e'][coord % 4]))
        plt.clf()
    assert np.all(p_values > 0.05)


if __name__ == "__main__":
    test_three_body()


# EOF
