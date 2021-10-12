import tensorflow as tf
import zfit
import zfit_physics as zphys


# TODO refactor these mass functions using e.g. a decorator.
#  Right now there is a lot of code repetition.
def gauss(mass, width):
    particle_mass = tf.cast(mass, tf.float64)
    particle_width = tf.cast(width, tf.float64)

    def mass_func(min_mass, max_mass, n_events):
        min_mass = tf.cast(min_mass, tf.float64)
        max_mass = tf.cast(max_mass, tf.float64)
        pdf = zfit.pdf.Gauss(mu=particle_mass, sigma=particle_width, obs="")
        iterator = tf.stack([min_mass, max_mass], axis=-1)
        return tf.vectorized_map(
            lambda lim: pdf.sample(1, limits=(lim[0], lim[1])), iterator
        )

    return mass_func


def breitwigner(mass, width):
    particle_mass = tf.cast(mass, tf.float64)
    particle_width = tf.cast(width, tf.float64)

    def mass_func(min_mass, max_mass, n_events):
        min_mass = tf.cast(min_mass, tf.float64)
        max_mass = tf.cast(max_mass, tf.float64)
        pdf = zfit.pdf.Cauchy(m=particle_mass, gamma=particle_width, obs="")
        iterator = tf.stack([min_mass, max_mass], axis=-1)
        return tf.vectorized_map(
            lambda lim: pdf.sample(1, limits=(lim[0], lim[1])), iterator
        )

    return mass_func


def relativistic_breitwigner(mass, width):
    particle_mass = tf.cast(mass, tf.float64)
    particle_width = tf.cast(width, tf.float64)

    def mass_func(min_mass, max_mass, n_events):
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

    return mass_func


_DEFAULT_CONVERTER = {
    "gauss": gauss,
    "bw": breitwigner,
    "relbw": relativistic_breitwigner,
}
