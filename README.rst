=====================
Tensorflow PhaseSpace
=====================

.. image:: https://travis-ci.org/zfit/tfphasespace.svg?branch=master
    :target: https://travis-ci.org/zfit/tfphasespace
.. image:: https://readthedocs.org/projects/tfphasespace/badge/?version=latest
   :target: https://tfphasespace.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

Python implementation of the Raubold and Lynch method for n-body events using
Tensorflow as a backend.

The code is based on the GENBOD function (W515 from CERNLIB), documented in

    F. James, Monte Carlo Phase Space, CERN 68-15 (1968)

and tries to follow it as closely as possible.

Detailed documentation, including the API, can be found in https://tfphasespace.readthedocs.io.

Why?
----
Lately, data analysis in High Energy Physics (HEP), traditionally performed within the `ROOT`_ ecosystem, has been moving more and more towards Python.
The possibility of carrying out purely Python-based analyses has become real thanks to the development of many open source Python packages,
which have allowed to replace most ROOT functionality with Python-based packages.

One of the aspects where this is still not possible is in the random generation of $n$-body phase space events, which are widely used in the field, for example to study kinematics
of the particle decays of interest, or to perform importance sampling in the case of complex amplitude models.
This has been traditionally done with the `TGenPhaseSpace`_ class, which is based of the GENBOD function of the CERNLIB FORTRAN libraries and which requires a full working ROOT installation. 

This package aims to address this issue by providing a Tensorflow-based implementation of such function to generate $n$-body decays without requiring a ROOT installation.
Additionally, an oft-needed functionality to generate complex decay chains, not included in ``TGenPhaseSpace``, is also offered, leaving room for decaying resonances (which don't have a fixed mass, but can be seen as a broad peak).

.. _ROOT: https://root.cern.ch
.. _TGenPhaseSpace: https://root.cern.ch/doc/master/classTGenPhaseSpace.html

Installing
----------

To install Tensorflow PhaseSpace, run this command in your terminal:

.. code-block:: console

    $ pip install tfphasespace

This is the preferred method to install Tensorflow PhaseSpace, as it will always install the most recent stable release.

For the newest development version (in case you really need it), you can install the version from git with

.. code-block:: console

   $ pip install git+https://github.com/zfit/tfphasespace


How to use
----------

The generation of simple n-body decays can be done using the `generate` function of `tfphasespace` with a 
very similar interface to ``TGenPhaseSpace``. For example, to generate :math:`B^0\to K\pi`, we would do::

   import tfphasespace
   import tensorflow as tf

   B0_MASS = 5279.58
   B0_AT_REST = [0.0, 0.0, 0.0, B0_MASS]
   PION_MASS = 139.57018
   KAON_MASS = 493.677

   weights, particles = tfphasespace.generate(B0_AT_REST,
                                              [PION_MASS, KAON_MASS],
                                              1000)

This generates tensorflow tensors, so no code has been executed yet. To run the Tensorflow graph, we simply do::

   weights, particles = tf.Session().run([weights, particles])

This returns an array of 1000 elements in the case of ``weights`` and a list of ``n_particles`` (2) arrays of $4\times1000$ shape,
where each of the dimensions corresponds to one of the components of the generated Lorentz 4-vector.

Sequential decays can be handled with the ``Particle`` class (used internally by ``generate``) and its `set_children` method.
As an example, to build the :math:`B^{0}\to K^{*}\gamma` decay in which :math:`K^*\to K\pi`, we would write::

   from tfphasespace import Particle
   import tensorflow as tf

   B0_MASS = 5279.58
   B0_AT_REST = [0.0, 0.0, 0.0, B0_MASS]
   KSTARZ_MASS = 895.81
   PION_MASS = 139.57018
   KAON_MASS = 493.677

   pion = Particle('pi+', PION_MASS)
   kaon = Particle('K+', KAON_MASS)
   kstar = Particle('K*', KSTARZ_MASS).set_children(pion, kaon)
   gamma = Particle('gamma', 0)
   bz = Particle('B0').set_children(kstar, gamma)

   with tf.Session() as sess:
      weights, particles = sess.run(bz.generate(B0_AT_REST, 1000))

Where we have used the fact that `set_children` returns the parent particle.
In this case, `particles` is a `dict` with the particle names as keys.
It is also important to note the mass is not necessary for the top particle, as it is determined
from the input 4-momentum.

More examples can be found in the `tests` folder and in the `documentation`_.

.. _documentation: https://tfphasespace.readthedocs.io/en/latest/usage.html


Physics validation
------------------

Physics validation is performed continuously in the included tests (``tests/test_physics.py``), run through Travis CI.
This validation is performed at two levels:

   + In simple $n$-body decays, the results of ``tfphasespace`` are checked against ``TGenPhaseSpace``.
   + For sequential decays, the results of ``tfphasespace`` are checked against `RapidSim`_, a "fast Monte Carlo generator for simulation of heavy-quark hadron decays".
      In the case of resonances, differences are expected because our tests don't include proper modelling of their mass shape, as it would require the introduction of 
      further dependencies. However, the results of the comparison can be expected visually.

The results of all physics validation performed by the ``tests_physics.py`` test are written in ``tests/plots``.

.. _RapidSim: https://github.com/gcowan/RapidSim/



Contributing
------------

Contributions are always welcome, please have a look at the `Contributing guide`_.

.. _Contributing guide: CONTRIBUTING.rst

