from __future__ import annotations

from collections.abc import Callable

import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from particle import Particle

import phasespace.fromdecay.mass_functions as mf

_kstarz = Particle.from_evtgen_name("K*0")
KSTARZ_MASS = _kstarz.mass
KSTARZ_WIDTH = _kstarz.width


def ref_mass_func(min_mass, max_mass, n_events):
    """Reference mass function used to compare the behavior of the actual mass functions.

    Args:
        min_mass: lower limit of mass.
        max_mass: upper limit of mass.
        n_events: number of mass values that should be generated.

    Returns:
        kstar_mass: Generated mass.

    Notes:
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
    "function",
    (mf.gauss_factory, mf.breitwigner_factory, mf.relativistic_breitwigner_factory),
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
    )  # Sample.shape have extra dimensions with just 1 or 0, which can be ignored
