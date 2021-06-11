=====
Usage
=====

The base of ``phasespace`` is the ``GenParticle`` object.
This object, which represents a particle, either stable or decaying, has only one mandatory argument, its name.

In most cases (except for the top particle of a decay), one wants to also specify its mass, which can be either
a number or ``tf.constant``, or a function.
Functions are used to specify the mass of particles such as resonances, which are not fixed but vary according to
a broad distribution.
These mass functions get three arguments, and must return a ``TensorFlow`` Tensor:

- The minimum mass allowed by the decay chain, which will be of shape `(n_events,)`.
- The maximum mass available, which will be of shape `(n_events,)`.
- The number of events to generate.

This function signature allows to handle threshold effects cleanly, giving enough information to produce kinematically
allowed decays (NB: ``phasespace`` will throw an error if a kinematically forbidden decay is requested).

A simple example
--------------------------


With these considerations in mind, one can build a decay chain by using the ``set_children`` method of the ``GenParticle``
class (which returns the class itself). As an example, to build the :math:`B^{0}\to K^{*}\gamma` decay in which
:math:`K^*\to K\pi` with a fixed mass, we would write:

.. jupyter-execute::

   from phasespace import GenParticle

   B0_MASS = 5279.58
   KSTARZ_MASS = 895.81
   PION_MASS = 139.57018
   KAON_MASS = 493.677

   pion = GenParticle('pi-', PION_MASS)
   kaon = GenParticle('K+', KAON_MASS)
   kstar = GenParticle('K*', KSTARZ_MASS).set_children(pion, kaon)
   gamma = GenParticle('gamma', 0)
   bz = GenParticle('B0', B0_MASS).set_children(kstar, gamma)

.. thebe-button:: Run this interactively


Phasespace events can be generated using the ``generate`` method, which gets the number of events to generate as input.
The method returns:

- The normalized weights of each event, as an array of dimension (n_events,).
- The 4-momenta of the generated particles as values of a dictionary with the particle name as key. These momenta
  are expressed as arrays of dimension (n_events, 4).

.. jupyter-execute::

   N_EVENTS = 1000

   weights, particles = bz.generate(n_events=N_EVENTS)

The ``generate`` method return an eager ``Tensor``: this is basically a wrapped numpy array. With ``Tensor.numpy()``,
it can always directly be converted to a numpy array (if really needed).

Boosting the particles
--------------------------


The particles are generated in the rest frame of the top particle.
To produce them at a given momentum of the top particle, one can pass these momenta with the ``boost_to`` argument in both
``generate`` and ~`tf.Tensor`. This latter approach can be useful if the momentum of the top particle
is generated according to some distribution, for example the kinematics of the LHC (see ``test_kstargamma_kstarnonresonant_lhc``
and ``test_k1gamma_kstarnonresonant_lhc`` in ``tests/test_physics.py`` to see how this could be done).

Weights
--------------------------


Additionally, it is possible to obtain the unnormalized weights by using the ``generate_unnormalized`` flag in
``generate``. In this case, the method returns the unnormalized weights, the per-event maximum weight
and the particle dictionary.

.. jupyter-execute::

    print(particles)


It is worth noting that the graph generation is cached even when using ``generate``, so iterative generation
can be performed using normal python loops without loss in performance:

.. jupyter-execute::

   for i in range(5):
       weights, particles = bz.generate(n_events=100)
       # ...
       # (do something with weights and particles)
       # ...



Resonances with variable mass
------------------------------


To generate the mass of a resonance, we need to give a function as its mass instead of a floating number.
This function should take as input the per-event lower mass allowed, per-event upper mass allowed and the number of
events, and should return a ~`tf.Tensor` with the generated masses and shape (nevents,). Well suited for this task
are the `TensorFlow Probability distributions <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions>`_
or, for more customized mass shapes, the
`zfit pdfs <https://zfit.github.io/zfit/model.html#tensor-sampling>`_ (currently an
*experimental feature* is needed, contact the `zfit developers <https://github.com/zfit/zfit>`_ to learn more).

Following with the same example as above, and approximating the resonance shape by a gaussian, we could
write the :math:`B^{0}\to K^{*}\gamma` decay chain as (more details can be found in ``tests/helpers/decays.py``):

.. jupyter-execute::
    :hide-output:

    import tensorflow as tf
    import tensorflow_probability as tfp
    from phasespace import GenParticle

    KSTARZ_MASS = 895.81
    KSTARZ_WIDTH = 47.4

    def kstar_mass(min_mass, max_mass, n_events):
       min_mass = tf.cast(min_mass, tf.float64)
       max_mass = tf.cast(max_mass, tf.float64)
       kstar_width_cast = tf.cast(KSTARZ_WIDTH, tf.float64)
       kstar_mass_cast = tf.cast(KSTARZ_MASS, dtype=tf.float64)

       kstar_mass = tf.broadcast_to(kstar_mass_cast, shape=(n_events,))
       if KSTARZ_WIDTH > 0:
           kstar_mass = tfp.distributions.TruncatedNormal(loc=kstar_mass,
                                                          scale=kstar_width_cast,
                                                          low=min_mass,
                                                          high=max_mass).sample()
       return kstar_mass

    bz = GenParticle('B0', B0_MASS).set_children(GenParticle('K*0', mass=kstar_mass)
                                                .set_children(GenParticle('K+', mass=KAON_MASS),
                                                              GenParticle('pi-', mass=PION_MASS)),
                                                GenParticle('gamma', mass=0.0))

    bz.generate(n_events=500)


Shortcut for simple decays
--------------------------

The generation of simple `n`-body decay chains can be done using the ``nbody_decay`` function of ``phasespace``, which takes

- The mass of the top particle.
- The mass of children particles as a list.
- The name of the top particle (optional).
- The names of the children particles (optional).

If the names are not given, `top` and `p_{i}` are assigned. For example, to generate :math:`B^0\to K\pi`, one would do:

.. jupyter-execute::

   import phasespace

   N_EVENTS = 1000

   B0_MASS = 5279.58
   PION_MASS = 139.57018
   KAON_MASS = 493.677

   decay = phasespace.nbody_decay(B0_MASS, [PION_MASS, KAON_MASS],
                                  top_name="B0", names=["pi", "K"])
   weights, particles = decay.generate(n_events=N_EVENTS)

In this example, ``decay`` is simply a ``GenParticle`` with the corresponding children.


Eager execution
---------------

By default, `phasespace` uses TensorFlow to build a graph of the computations. This is usually more
performant, especially if used multiple times. However, this has the disadvantage that _inside_
`phasespac`, the actual values are not computed on Python runtime, e.g. if a breakpoint is set
the values of a `tf.Tensor` won't be available.

TensorFlow (since version 2.0) however can easily switch to so called "eager execution": in this
mode, it behaves the same as Numpy; values are computed instantly and the Python code is not only
executed once but every time.

To switch this on or off, the global flag in TensorFlow `tf.config.run_functions_eagerly(True)` or
the enviroment variable "PHASESPACE_EAGER" (which switches this flag) can be used.

Random numbers
--------------

The random number generation inside `phasespace` is transparent in order to allow for deterministic
behavior if desired. A function that uses random number generation inside always takes a `seed` (or `rng`)
argument. The behavior is as follows

- if no seed is given, the global random number generator of TensorFlow will be used. Setting this
  instance explicitly or by setting the seed via `tf.random.set_seed` allows for a deterministic
  execution of a whole _script_.
- if the seed is a number it will be used to create a random number generator from this. Using the
  same seed again will result in the same output.
- if the seed is an instance of :py:class:`tf.random.Generator`, this instance will directly be used
  and advances an undefined number of steps.
