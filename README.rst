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
A statement of need


Installing
----------

To install Tensorflow PhaseSpace, run this command in your terminal:

.. code-block:: console

    $ pip install tfphasespace

This is the preferred method to install Tensorflow PhaseSpace, as it will always install the most recent stable release.

You can also install from the git source by cloning the public repository

.. code-block:: console

    $ git clone git://github.com/zfit/tfphasespace

or downloading the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/zfit/tfphasespace/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/zfit/tfphasespace
.. _tarball: https://github.com/zfit/tfphasespace/tarball/master


How to use
----------


Physics validation
------------------
In addition, this package adds the possibility of generating sequential decays including
non-fixed-mass particles, ie, resonances.


Performance
-----------

Contributing
------------

Contributions are always welcome, please have a look at the `Contributing guide`_.

.. _Contributing guide: CONTRIBUTING.rst

