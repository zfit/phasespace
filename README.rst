==========
PhaseSpace
==========

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2591993.svg
   :target: https://doi.org/10.5281/zenodo.2591993
.. image:: https://img.shields.io/pypi/status/phasespace.svg
   :target: https://pypi.org/project/phasespace/
.. image:: https://img.shields.io/pypi/pyversions/phasespace.svg
   :target: https://pypi.org/project/phasespace/
.. image:: https://travis-ci.org/zfit/phasespace.svg?branch=master
   :target: https://travis-ci.org/zfit/phasespace
.. image:: https://coveralls.io/repos/github/zfit/phasespace/badge.svg?branch=master
    :target: https://coveralls.io/github/zfit/phasespace?branch=master
.. image:: https://readthedocs.org/projects/phasespace/badge/?version=stable
   :target: https://phasespace.readthedocs.io/en/latest/?badge=stable
   :alt: Documentation Status
.. image:: https://badges.gitter.im/zfit/phasespace.svg
   :target: https://gitter.im/zfit/phasespace?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
   :alt: Gitter chat
   
Python implementation of the Raubold and Lynch method for `n`-body events using
TensorFlow as a backend.

The code is based on the GENBOD function (W515 from CERNLIB), documented in [1]
and tries to follow it as closely as possible.

Detailed documentation, including the API, can be found in https://phasespace.readthedocs.io.
Don't hesitate to join our `gitter`_ channel for questions and comments.

If you use phasespace in a scientific publication we would appreciate citations to the `zenodo`_ publication:

.. code-block:: bibtex

   @article{phasespace-2019,
    title={phasespace: n-body phase space generation in Python},
    DOI={10.5281/zenodo.2926058},
    publisher={Zenodo},
    author={Albert Puig and Jonas Eschle},
    year={2019},
    month={Mar}}

Free software: BSD-3-Clause.

[1]  F. James, Monte Carlo Phase Space, CERN 68-15 (1968)

.. _zenodo: https://doi.org/10.5281/zenodo.2591993
.. _Gitter: https://gitter.im/zfit/phasespace


Why?
----
Lately, data analysis in High Energy Physics (HEP), traditionally performed within the `ROOT`_ ecosystem, has been moving more and more towards Python.
The possibility of carrying out purely Python-based analyses has become real thanks to the development of many open source Python packages,
which have allowed to replace most ROOT functionality with Python-based packages.

One of the aspects where this is still not possible is in the random generation of `n`-body phase space events, which are widely used in the field, for example to study kinematics
of the particle decays of interest, or to perform importance sampling in the case of complex amplitude models.
This has been traditionally done with the `TGenPhaseSpace`_ class, which is based of the GENBOD function of the CERNLIB FORTRAN libraries and which requires a full working ROOT installation.

This package aims to address this issue by providing a TensorFlow-based implementation of such function to generate `n`-body decays without requiring a ROOT installation.
Additionally, an oft-needed functionality to generate complex decay chains, not included in ``TGenPhaseSpace``, is also offered, leaving room for decaying resonances (which don't have a fixed mass, but can be seen as a broad peak).

.. _ROOT: https://root.cern.ch
.. _TGenPhaseSpace: https://root.cern.ch/doc/master/classTGenPhaseSpace.html

Installing
----------

To install ``phasespace``, run this command in your terminal:

.. code-block:: console

    $ pip install phasespace

This is the preferred method to install ``phasespace``, as it will always install the most recent stable release.

For the newest development version, which may be unstable, you can install the version from git with

.. code-block:: console

   $ pip install git+https://github.com/zfit/phasespace


How to use
----------

The generation of simple `n`-body decays can be done using the ``generate`` function of ``phasespace`` with a
very simple interface: one needs to pass the mass of the top particle and the masses of the children particle as
a list. For example, to generate :math:`B^0\to K\pi`, we would do:

.. code-block:: python

   import phasespace

   B0_MASS = 5279.58
   PION_MASS = 139.57018
   KAON_MASS = 493.677

   weights, particles = phasespace.generate(B0_MASS,
                                            [PION_MASS, KAON_MASS],
                                            n_events=1000)

This returns a numpy array of 1000 elements in the case of ``weights`` and a list of ``n particles`` (2) arrays of (1000, 4) shape,
where each of the 4-dimensions corresponds to one of the components of the generated Lorentz 4-vector.
All particles are generated in the rest frame of the top particle; boosting to a certain momentum (or list of momenta) can be
achieved by passing the momenta to the ``boost_to`` argument.

Behind the scenes, this function runs the TensorFlow graph, but no caching of the graph or reusing the session is performed.
If we want to get the graph to avoid an immediate execution, we can use the `as_numpy` flag. Then, to produce the equivalent result
to the previous example, we simply do:

.. code-block:: python

   import tensorflow as tf

   with tf.Session() as sess:
       weights, particles = phasespace.generate(B0_MASS,
                                                [PION_MASS, KAON_MASS],
                                                n_events=1000,
                                                as_numpy=False)
       weights, particles = sess.run([weights, particles])

Sequential decays can be handled with the ``Particle`` class (used internally by ``generate``) and its ``set_children`` method.
As an example, to build the :math:`B^{0}\to K^{*}\gamma` decay in which :math:`K^*\to K\pi`, we would write:

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

   weights, particles = bz.generate(n_events=1000)

Where we have used the fact that ``set_children`` returns the parent particle.
In this case, ``particles`` is a ``dict`` with the particle names as keys:

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

The `Particle` class is able to cache the graphs so it is possible to generate in a loop
without overhead:

.. code-block:: pycon

    for i in range(10):
        weights, particles = bz.generate(n_events=1000)
        ...
        (do something with weights and particles)
        ...

This way of generating is recommended in the case of large samples, as it allows to benefit from
parallelisation while at the same time keep the memory usage low.

If we want to operate with the TensorFlow graph instead, we can use the `generate_tensor` method
of `Particle`, which has the same signature as `generate`.

More examples can be found in the ``tests`` folder and in the `documentation`_.

.. _documentation: https://phasespace.readthedocs.io/en/latest/usage.html


Physics validation
------------------

Physics validation is performed continuously in the included tests (``tests/test_physics.py``), run through Travis CI.
This validation is performed at two levels:

- In simple `n`-body decays, the results of ``phasespace`` are checked against ``TGenPhaseSpace``.
- For sequential decays, the results of ``phasespace`` are checked against `RapidSim`_, a "fast Monte Carlo generator for simulation of heavy-quark hadron decays".
  In the case of resonances, differences are expected because our tests don't include proper modelling of their mass shape, as it would require the introduction of
  further dependencies. However, the results of the comparison can be expected visually.

The results of all physics validation performed by the ``tests_physics.py`` test are written in ``tests/plots``.

.. _RapidSim: https://github.com/gcowan/RapidSim/



Contributing
------------

Contributions are always welcome, please have a look at the `Contributing guide`_.

.. _Contributing guide: CONTRIBUTING.rst

