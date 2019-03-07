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
import subprocess

import numpy as np
from scipy.stats import ks_2samp

if platform.system() == 'Darwin':
    import matplotlib

    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import uproot

import tensorflow as tf

from tfphasespace import tfphasespace

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PLOT_DIR = os.path.join(BASE_PATH, 'tests', 'plots')

B_MASS = 5279.0
B_AT_REST = tf.stack((0.0, 0.0, 0.0, B_MASS), axis=-1)
PION_MASS = 139.6


def make_norm_histo(array, range_, weights=None):
    """Make histo and modify dimensions."""
    histo = np.histogram(array, 100, range=range_, weights=weights)[0]
    return histo / np.sum(histo)


def create_ref_histos(n_pions):
    """Load reference histogram data."""

    def make_histos(vector_list, range_, weights=None):
        """Make histograms."""
        v_array = np.stack([vector_list.x, vector_list.y, vector_list.z, vector_list.E])
        histos = tuple(np.histogram(v_array[coord], 100, range=range_, weights=weights)[0]
                       for coord in range(4))
        return tuple(histo / np.sum(histo) for histo in histos)

    ref_dir = os.path.join(BASE_PATH, 'data')
    if not os.path.exists(ref_dir):
        os.mkdir(ref_dir)
    ref_file = os.path.join(ref_dir, 'bto{}pi.root'.format(n_pions))
    if not os.path.exists(ref_file):
        script = os.path.join(BASE_PATH,
                              'scripts',
                              'prepare_test_samples.cxx+({})'
                              .format(','.join(['"{}"'.format(os.path.join(BASE_PATH,
                                                                           'data',
                                                                           'bto{}pi.root'.format(i + 1)))
                                                for i in range(1, 4)])))
        subprocess.call("echo $PATH", shell=True)
        subprocess.call("root -qb '{}'".format(script), shell=True)
    events = uproot.open(ref_file)['events']
    pion_names = ['pion_{}'.format(pion + 1) for pion in range(n_pions)]
    pions = {pion_name: events.array(pion_name)
             for pion_name in pion_names}
    weights = events.array('weight')
    return [make_norm_histo(array,
                            range_=(-3000 if coord % 4 != 3 else 0, 3000),
                            weights=weights)
            for pion in pions.values()
            for coord, array in enumerate([pion.x, pion.y, pion.z, pion.E])], \
           make_norm_histo(weights, range_=(0, 1))


def run_test(n_particles, test_prefix):
    sess = tf.Session()
    weights, particles = sess.run(tfphasespace.generate(B_AT_REST,
                                                        [PION_MASS] * n_particles,
                                                        100000))
    parts = np.concatenate(particles, axis=0)
    histos = [make_norm_histo(parts[coord],
                              range_=(-3000 if coord % 4 != 3 else 0, 3000),
                              weights=weights)
              for coord in range(parts.shape[0])]
    weight_histos = make_norm_histo(weights, range_=(0, 1))
    ref_histos, ref_weights = create_ref_histos(n_particles)
    p_values = np.array([ks_2samp(histos[coord], ref_histos[coord])[1]
                         for coord, _ in enumerate(histos)] +
                        [ks_2samp(weight_histos, ref_weights)[1]])
    # Let's plot
    x = np.linspace(-3000, 3000, 100)
    e = np.linspace(0, 3000, 100)
    if not os.path.exists(PLOT_DIR):
        os.mkdir(PLOT_DIR)
    for coord, _ in enumerate(histos):
        plt.hist(x if coord % 4 != 3 else e, weights=histos[coord], alpha=0.5, label='tfphasespace', bins=100)
        plt.hist(x if coord % 4 != 3 else e, weights=ref_histos[coord], alpha=0.5, label='TGenPhasespace', bins=100)
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(PLOT_DIR,
                                 "{}_pion_{}_{}.png".format(test_prefix,
                                                            int(coord / 4) + 1,
                                                            ['x', 'y', 'z', 'e'][coord % 4])))
        plt.clf()
    plt.hist(np.linspace(0, 1, 100), weights=weight_histos, alpha=0.5, label='tfphasespace', bins=100)
    plt.hist(np.linspace(0, 1, 100), weights=ref_weights, alpha=0.5, label='tfphasespace', bins=100)
    plt.savefig(os.path.join(PLOT_DIR, '{}_weights.png'.format(test_prefix)))
    plt.clf()
    assert np.all(p_values > 0.05)
    sess.close()


def test_two_body():
    """Test B->pipi decay."""
    run_test(2, "two_body")


def test_three_body():
    """Test B -> pi pi pi decay."""
    run_test(3, "three_body")


def test_four_body():
    """Test B -> pi pi pi pi decay."""
    run_test(4, "four_body")


if __name__ == "__main__":
    test_two_body()
    test_three_body()
    test_four_body()

# EOF
