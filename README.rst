=====================
TensorFlow PhaseSpace
=====================

.. image:: https://zenodo.org/badge/172891230.svg
   :target: https://zenodo.org/badge/latestdoi/172891230
.. image:: https://img.shields.io/pypi/status/phasespace.svg
   :target: https://pypi.org/project/phasespace/
.. image:: https://img.shields.io/pypi/pyversions/phasespace.svg
   :target: https://pypi.org/project/phasespace/
.. image:: https://travis-ci.org/zfit/phasespace.svg?branch=master
   :target: https://travis-ci.org/zfit/phasespace
.. image:: https://readthedocs.org/projects/phasespace/badge/?version=stable
   :target: https://phasespace.readthedocs.io/en/latest/?badge=stable
   :alt: Documentation Status
.. image:: https://badges.gitter.im/zfit/phasespace.svg
   :alt: Join the chat at https://gitter.im/zfit/phasespace
   :target: https://gitter.im/zfit/phasespace?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

Python implementation of the Raubold and Lynch method for `n`-body events using
TensorFlow as a backend.

The code is based on the GENBOD function (W515 from CERNLIB), documented in [1]
and tries to follow it as closely as possible.

Detailed documentation, including the API, can be found in https://phasespace.readthedocs.io.

Free software: BSD-3-Clause.

[1]  F. James, Monte Carlo Phase Space, CERN 68-15 (1968)

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

To install TensorFlow PhaseSpace, run this command in your terminal:

.. code-block:: console

    $ pip install phasespace

This is the preferred method to install TensorFlow PhaseSpace, as it will always install the most recent stable release.

For the newest development version (in case you really need it), you can install the version from git with

.. code-block:: console

   $ pip install git+https://github.com/zfit/phasespace


How to use
----------

The generation of simple `n`-body decays can be done using the ``generate`` function of ``phasespace`` with a
very similar interface to ``TGenPhaseSpace``. For example, to generate :math:`B^0\to K\pi`, we would do:

.. code-block:: python

   import phasespace
   import tensorflow as tf

   B0_MASS = 5279.58
   B0_AT_REST = [0.0, 0.0, 0.0, B0_MASS]
   PION_MASS = 139.57018
   KAON_MASS = 493.677

   weights, particles = phasespace.generate(B0_AT_REST,
                                            [PION_MASS, KAON_MASS],
                                            1000)

This generates TensorFlow tensors, so no code has been executed yet. To run the TensorFlow graph, we simply do:

.. code-block:: python

   with tf.Session() as sess:
      weights, particles = sess.run([weights, particles])

This returns an array of 1000 elements in the case of ``weights`` and a list of `n particles` (2) arrays of (4, 1000) shape,
where each of the 4-dimensions corresponds to one of the components of the generated Lorentz 4-vector.

Sequential decays can be handled with the ``Particle`` class (used internally by ``generate``) and its ``set_children`` method.
As an example, to build the :math:`B^{0}\to K^{*}\gamma` decay in which :math:`K^*\to K\pi`, we would write:

.. code-block:: python

   from phasespace import Particle
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

Where we have used the fact that ``set_children`` returns the parent particle.
In this case, ``particles`` is a ``dict`` with the particle names as keys:

.. code-block:: pycon

   >>> particles
   {'K*': array([[-2259.88717495,   742.20158838, -1419.57804967, ...,
            385.51632682,   890.89417859, -1938.80489221],
         [ -491.3119786 , -2348.67021741, -2049.19459865, ...,
            -932.58261761, -1054.16217965, -1669.40481126],
         [-1106.5946257 ,   711.27644522,  -598.85626591, ...,
         -2356.84025605, -2160.57372728,  -164.77965753],
         [ 2715.78804872,  2715.78804872,  2715.78804872, ...,
            2715.78804872,  2715.78804872,  2715.78804872]]),
   'K+': array([[-1918.74294565,   363.10302225,  -830.13803095, ...,
               9.28960349,   850.87382095,  -895.29815921],
         [ -566.15415012,  -956.94044749, -1217.14751182, ...,
            -243.52446264, -1095.04308712, -1078.03237584],
         [-1108.26109897,   534.79579335,  -652.41135612, ...,
            -901.56453631, -2069.39723754,  -244.1159568 ],
         [ 2339.67191226,  1255.90698132,  1685.21060224, ...,
            1056.37401241,  2539.53293518,  1505.66336806]]),
   'gamma': array([[2259.88717495, -742.20158838, 1419.57804967, ..., -385.51632682,
         -890.89417859, 1938.80489221],
         [ 491.3119786 , 2348.67021741, 2049.19459865, ...,  932.58261761,
         1054.16217965, 1669.40481126],
         [1106.5946257 , -711.27644522,  598.85626591, ..., 2356.84025605,
         2160.57372728,  164.77965753],
         [2563.79195128, 2563.79195128, 2563.79195128, ..., 2563.79195128,
         2563.79195128, 2563.79195128]]),
   'pi+': array([[ -341.14422931,   379.09856613,  -589.44001872, ...,
            376.22672333,    40.02035764, -1043.506733  ],
         [   74.84217153, -1391.72976992,  -832.04708683, ...,
            -689.05815497,    40.88090746,  -591.37243542],
         [    1.66647327,   176.48065186,    53.55509021, ...,
         -1455.27571974,   -91.17648974,    79.33629927],
         [  376.11613646,  1459.8810674 ,  1030.57744648, ...,
            1659.41403631,   176.25511354,  1210.12468065]])}

It is also important to note the mass is not necessary for the top particle, as it is determined
from the input 4-momentum.

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

