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
Additionally, an oft-needed functionality to generate complex decay chains, not included in TGenPhaseSpace, is also offered, leaving room for decaying resonances (which don't have a fixed mass, but can be seen as a broad peak).

.. _ROOT: https://root.cern.ch
.. _TGenPhaseSpace: https://root.cern.ch/doc/v610/classTGenPhaseSpace.html

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


Physics validation
------------------
In addition, this package adds the possibility of generating sequential decays including
non-fixed-mass particles, *i.e.*, resonances.


Performance
-----------

Contributing
------------

Contributions are always welcome, please have a look at the `Contributing guide`_.

.. _Contributing guide: CONTRIBUTING.rst

