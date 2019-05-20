=====
Usage
=====

The base of ``phasespace`` is the ``Particle`` object.
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

With these considerations in mind, one can build a decay chain by using the ``set_children`` method of the ``Particle``
class (which returns the class itself). As an example, to build the :math:`B^{0}\to K^{*}\gamma` decay in which
:math:`K^*\to K\pi` with a fixed mass, we would write:

.. code-block:: python

   from phasespace import Particle

   B0_MASS = 5279.58
   KSTARZ_MASS = 895.81
   PION_MASS = 139.57018
   KAON_MASS = 493.677

   pion = Particle('pi+', PION_MASS)
   kaon = Particle('K+', KAON_MASS)
   kstar = Particle('K*', KSTARZ_MASS).set_children(pion, kaon)
   gamma = Particle('gamma', 0)
   bz = Particle('B0', B0_MASS).set_children(kstar, gamma)

Phasespace events can be generated using the ``generate`` method, which gets the number of events to generate as input.
The method returns:

- The normalized weights of each event, as an array of dimension (n_events,).
- The 4-momenta of the generated particles as values of a dictionary with the particle name as key. These momenta
  are expressed as arrays of dimension (n_events, 4).

.. code-block:: python

   N_EVENTS = 1000

   weights, particles = bz.generate(n_events=N_EVENTS)

The ``generate`` method directly produces numpy arrays; for advanced usage, ``generate_tensor`` returns the same objects with the
numpy arrays replaced by ``tf.Tensor`` of the same shape.
So one can do, equivalent to the previous example:

.. code-block:: python

   import tensorflow as tf

   with tf.Session() as sess:
       weights, particles = sess.run(bz.generate_tensor(n_events=N_EVENTS))

In both cases, the particles are generated in the rest frame of the top particle.
To produce them at a given momentum of the top particle, one can pass these momenta with the ``boost_to`` argument in both
``generate`` and ``generate_tensor``. This latter approach can be useful if the momentum of the top particle
is generated according to some distribution, for example the kinematics of the LHC (see ``test_kstargamma_kstarnonresonant_lhc``
and ``test_k1gamma_kstarnonresonant_lhc`` in ``tests/test_physics.py`` to see how this could be done).

Additionally, it is possible to obtain the unnormalized weights by using the ``generate_unnormalized`` flag in  
``generate`` and ``generate_tensor``. In this case, the method returns the unnormalized weights, the per-event maximum weight
and the particle dictionary.

.. code-block:: pycon

   >>> particles
   {'K*': array([[ 1732.79325872, -1632.88873127,   950.85807735,  2715.78804872],
          [-1633.95329448,   239.88921123, -1961.0402768 ,  2715.78804872],
          [  407.15613764, -2236.6569286 , -1185.16616251,  2715.78804872],
          ...,
          [ 1091.64603395, -1301.78721269,  1920.07503991,  2715.78804872],
          [ -517.3125083 ,  1901.39296899,  1640.15905194,  2715.78804872],
          [  656.56413668,  -804.76922982,  2343.99214816,  2715.78804872]]),
    'K+': array([[  750.08077976,  -547.22569019,   224.6920906 ,  1075.30490935],
          [-1499.90049089,   289.19714633, -1935.27960292,  2514.43047106],
          [   97.64746732, -1236.68112923,  -381.09526192,  1388.47607911],
          ...,
          [  508.66157459,  -917.93523639,  1474.7064148 ,  1876.11771642],
          [ -212.28646168,   540.26381432,   610.86656669,   976.63988936],
          [  177.16656666,  -535.98777569,   946.12636904,  1207.28744488]]),
    'gamma': array([[-1732.79325872,  1632.88873127,  -950.85807735,  2563.79195128],
          [ 1633.95329448,  -239.88921123,  1961.0402768 ,  2563.79195128],
          [ -407.15613764,  2236.6569286 ,  1185.16616251,  2563.79195128],
          ...,
          [-1091.64603395,  1301.78721269, -1920.07503991,  2563.79195128],
          [  517.3125083 , -1901.39296899, -1640.15905194,  2563.79195128],
          [ -656.56413668,   804.76922982, -2343.99214816,  2563.79195128]]),
    'pi+': array([[  982.71247896, -1085.66304109,   726.16598675,  1640.48313937],
          [ -134.0528036 ,   -49.3079351 ,   -25.76067389,   201.35757766],
          [  309.50867032,  -999.97579937,  -804.0709006 ,  1327.31196961],
          ...,
          [  582.98445936,  -383.85197629,   445.36862511,   839.6703323 ],
          [ -305.02604662,  1361.12915468,  1029.29248526,  1739.14815935],
          [  479.39757002,  -268.78145413,  1397.86577911,  1508.50060384]])}

It is worth noting that the graph generation is cached even when using ``generate``, so iterative generation
can be performed using normal python loops without loss in performance:

.. code-block:: python

   for i in range(10):
       weights, particles = bz.generate(n_events=1000)
       ...
       (do something with weights and particles)
       ...

To generate the mass of a resonance, we need to give a function as its mass instead of a floating number.
This function should take as input the per-event lower mass allowed, per-event upper mass allowed and the number of
events, and should return a `tf.Tensor` with the generated masses and shape (nevents,). Well suited for this task
are the `TensorFlow Probability distributions <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions>`_
or, for more customized mass shapes, the
`zfit pdfs <https://zfit.github.io/zfit/model.html#tensor-sampling>`_ *(currently an
experimental feature is needed, contact the `zfit developers <https://github.com/zfit/zfit>`_ to learn more).*

Following with the same example as above, and approximating the resonance shape by a gaussian, we could
write the :math:`B^{0}\to K^{*}\gamma` decay chain as (more details can be found in ``tests/helpers/decays.py``):

.. code-block:: python

   import tensorflow as tf
   import tensorflow_probability as tfp
   from phasespace import Particle

   KSTARZ_MASS = 895.81
   KSTARZ_WIDTH = 47.4

     def kstar_mass(min_mass, max_mass, n_events):
        min_mass = tf.cast(min_mass, tf.float64)
        max_mass = tf.cast(max_mass, tf.float64)
        kstar_width_cast = tf.cast(KSTARZ_WIDTH, tf.float64)
        kstar_mass_cast = tf.cast(KSTARZ_MASS, dtype=tf.float64)

        kstar_mass = tf.broadcast_to(kstar_mass_cast, shape=(n_events,))
        if kstar_width > 0:
            kstar_mass = tfp.distributions.TruncatedNormal(loc=kstar_mass,
                                                           scale=kstar_width_cast,
                                                           low=min_mass,
                                                           high=max_mass).sample()
        return kstar_mass

   bz = Particle('B0', B0_MASS).set_children(Particle('K*0', mass=kstar_mass)
                                             .set_children(Particle('K+', mass=KAON_MASS),
                                                           Particle('pi-', mass=PION_MASS)),
                                             Particle('gamma', mass=0.0))


Shortcut for simple decays
--------------------------

The generation of simple `n`-body decays can be done using the ``generate`` function of ``phasespace``, which takes

- The mass of the top particle.
- The mass of children particles as a list.
- The number of events to generate.
- The optional ``boost_to`` tensor.

For example, to generate :math:`B^0\to K\pi`, one would do:

.. code-block:: python

   import phasespace

   N_EVENTS = 1000

   B0_MASS = 5279.58
   PION_MASS = 139.57018
   KAON_MASS = 493.677

   weights, particles = phasespace.generate(B0_MASS,
                                            [PION_MASS, KAON_MASS],
                                            n_events=N_EVENTS)


Internally, this function builds a decay chain using ``Particle``, and therefore the same considerations as before apply.
To avoid running the TensorFlow graph, one can set the ``as_numpy`` flag to ``False`` to get the graphs instead of the
numpy arrays.
