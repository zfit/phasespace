"""Mass distribution functions for resonant particles.

This module provides factory functions that create mass distribution functions for resonant particles. These
functions use zfit PDFs to sample masses from various distributions (Gaussian, Breit-Wigner, etc.) within
specified limits.
"""

import tensorflow as tf
import zfit
import zfit_physics as zphys

# TODO refactor these mass functions using e.g. a decorator.
#  Right now there is a lot of code repetition.


def gauss_factory(mass, width):
    """Create a Gaussian mass distribution function.

    Args:
        mass: Mean mass of the particle.
        width: Width (sigma) of the Gaussian distribution.

    Returns:
        Callable that generates masses from a Gaussian distribution.
        The returned function accepts ``min_mass``, ``max_mass``, and ``n_events``
        parameters and returns sampled masses as a tensor of shape ``(n_events,)``.
    """
    particle_mass = tf.cast(mass, tf.float64)
    particle_width = tf.cast(width, tf.float64)

    def gauss(min_mass, max_mass, n_events):
        min_mass = tf.cast(min_mass, tf.float64)
        max_mass = tf.cast(max_mass, tf.float64)
        pdf = zfit.pdf.Gauss(mu=particle_mass, sigma=particle_width, obs="")
        iterator = tf.stack([min_mass, max_mass], axis=-1)
        return tf.vectorized_map(
            lambda lim: pdf.sample(1, limits=(lim[0], lim[1])).unstack_x(), iterator
        )

    return gauss


def breitwigner_factory(mass, width):
    """Create a Breit-Wigner (Cauchy) mass distribution function.

    Args:
        mass: Central mass (m) of the particle.
        width: Width (gamma) of the Breit-Wigner distribution.

    Returns:
        Callable that generates masses from a Breit-Wigner distribution.
        The returned function accepts ``min_mass``, ``max_mass``, and ``n_events``
        parameters and returns sampled masses as a tensor of shape ``(n_events,)``.
    """
    particle_mass = tf.cast(mass, tf.float64)
    particle_width = tf.cast(width, tf.float64)

    def bw(min_mass, max_mass, n_events):
        min_mass = tf.cast(min_mass, tf.float64)
        max_mass = tf.cast(max_mass, tf.float64)
        pdf = zfit.pdf.Cauchy(m=particle_mass, gamma=particle_width, obs="")
        iterator = tf.stack([min_mass, max_mass], axis=-1)
        return tf.vectorized_map(
            lambda lim: pdf.sample(1, limits=(lim[0], lim[1])).unstack_x(), iterator
        )

    return bw


def relativistic_breitwigner_factory(mass, width):
    """Create a relativistic Breit-Wigner mass distribution function.

    Args:
        mass: Central mass (m) of the particle.
        width: Width (gamma) of the relativistic Breit-Wigner distribution.

    Returns:
        Callable that generates masses from a relativistic Breit-Wigner distribution.
        The returned function accepts ``min_mass``, ``max_mass``, and ``n_events``
        parameters and returns sampled masses as a tensor of shape ``(n_events,)``.

    Notes:
        This uses ``tf.map_fn`` instead of ``tf.vectorized_map`` as no analytic
        sampling is available for the relativistic Breit-Wigner distribution.
    """
    particle_mass = tf.cast(mass, tf.float64)
    particle_width = tf.cast(width, tf.float64)

    def relbw(min_mass, max_mass, n_events):
        min_mass = tf.cast(min_mass, tf.float64)
        max_mass = tf.cast(max_mass, tf.float64)
        pdf = zphys.pdf.RelativisticBreitWigner(
            m=particle_mass, gamma=particle_width, obs=""
        )
        iterator = tf.stack([min_mass, max_mass], axis=-1)

        # this works with map_fn but not with vectorized_map as no analytic sampling is available.
        return tf.map_fn(
            lambda lim: pdf.sample(1, limits=(lim[0], lim[1])).unstack_x(), iterator
        )

    return relbw


DEFAULT_CONVERTER = {
    "gauss": gauss_factory,
    "bw": breitwigner_factory,
    "relbw": relativistic_breitwigner_factory,
}
