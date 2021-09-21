from typing import Callable

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import phasespace.fulldecay.mass_functions as mf

KSTARZ_MASS = 895.81
KSTARZ_WIDTH = 47.4


def ref_mass_func(min_mass, max_mass, n_events):
    """Reference mass function used to compare the behavior of the actual mass functions.

    Parameters
    ----------
    min_mass
    max_mass
    n_events

    Returns
    -------
    kstar_mass
        Mass generated

    Notes
    -----
    Code taken from phasespace documentation.
    """
    min_mass = tf.cast(min_mass, tf.float64)
    max_mass = tf.cast(max_mass, tf.float64)
    kstar_width_cast = tf.cast(KSTARZ_WIDTH, tf.float64)
    kstar_mass_cast = tf.cast(KSTARZ_MASS, dtype=tf.float64)
    kstar_mass = tf.broadcast_to(kstar_mass_cast, shape=(n_events,))
    kstar_mass = tfp.distributions.TruncatedNormal(
        loc=kstar_mass, scale=kstar_width_cast, low=min_mass, high=max_mass
    ).sample()
    return kstar_mass


@pytest.mark.parametrize(
    "function", (mf.gauss, mf.breitwigner, mf.relativistic_breitwigner)
)
@pytest.mark.parametrize("size", (1, 10))
def test_shape(function: Callable, size: int, params: tuple = (1.0, 1.0)):
    rng = tf.random.Generator.from_seed(1234)
    min_max_mass = rng.uniform(minval=0, maxval=1000, shape=(2, size), dtype=tf.float64)
    min_mass, max_mass = tf.unstack(tf.sort(min_max_mass, axis=0), axis=0)
    assert tf.reduce_all(min_mass <= max_mass)
    ref_sample = ref_mass_func(min_mass, max_mass, len(min_mass))
    sample = function(*params)(min_mass, max_mass, len(min_mass))
    assert sample.shape[0] == ref_sample.shape[0]
    assert all(
        i <= 1 for i in sample.shape[1:]
    )  # Sample.shape have extra dimensions with just 1 or 0, but code still seems to work
