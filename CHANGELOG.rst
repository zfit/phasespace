*********
Changelog
*********

Develop
==========


Major Features and Improvements
-------------------------------

Behavioral changes
------------------


Bug fixes and small changes
---------------------------

Requirement changes
-------------------


Thanks
------

1.8.0 (27 Jan 2023)
===================

Requirement changes
-------------------
- upgrade to zfit >= 0.10.0 and zfit-physics >= 0.3.0
- pinning uproot and awkward to ~4 and ~1, respectively


1.7.0 (1. Sep 2022)
====================

Upgraded Python and TensorFlow version.

Added ``tf`` and ``tensorflow`` extra to requirements. If you intend to use
phasespace with TensorFlow in the future (and not another backend like numpy or JAX),
make sure to always install with ``phasespace[tf]``.

Requirement changes
-------------------
- upgrade to TensorFlow >= 2.7
- Python from 3.7 to 3.10 is now supported

1.6.0 (14 Apr 2022)
====================

Major Features and Improvements
-------------------------------
- Improved GenMultiDecay to have better control on the decay mass of non-stable particles.
- Added a `particle_model_map` argument to the `GenMultiDecay` class. This is a
  dict where the key is a particle name and the value is a mass function name.
  The feature can be seen in the
  `GenMultiDecay Tutorial <https://github.com/zfit/phasespace/blob/master/docs/GenMultiDecay_Tutorial.ipynb>`_.


1.5.0 (27 Nov 2021)
===================


Major Features and Improvements
-------------------------------
- add support to generate from a DecayChain using
  `the decaylanguage <https://github.com/scikit-hep/decaylanguage>`_ package from Scikit-HEP.
  This is in the new subpackage "fromdecay" and can be used by installing the extra with
  ``pip install phasespace[fromdecay]``.


Requirement changes
-------------------
- drop Python 3.6 support


Thanks
------
- to Simon Thor for contributing the ``fromdecay`` subpackage.

1.4.2 (5.11.2021)
==================

Requirement changes
-------------------
- Losen restriction on TensorFlow, allow version 2.7 (and 2.5, 2.6)

1.4.1 (27.08.2021)
==================

Requirement changes
-------------------
- Losen restriction on TensorFlow, allow version 2.6 (and 2.5)

1.4.0 (11.06.2021)
==================

Requirement changes
-------------------
- require TensorFlow 2.5 as 2.4 breaks some functionality

1.3.0 (28.05.2021)
===================


Major Features and Improvements
-------------------------------

- Support Python 3.9
- Support TensorFlow 2.5
- improved compilation in tf.functions, use of XLA where applicable
- developer: modernization of setup, CI and more

Thanks
------

- Remco de Boer for many commits and cleanups

1.2.0 (17.12.20)
================


Major Features and Improvements
-------------------------------

- Python 3.8 support
- Allow eager execution by setting with `tf.config.run_functions_eagerly(True)`
  or the environment variable "PHASESPACE_EAGER"
- Deterministic random number generation via seed
  or `tf.random.Generator` instance

Behavioral changes
------------------


Bug fixes and small changes
---------------------------

Requirement changes
-------------------

- tighten TensorFlow to 2.3/2.4
- tighten TensorFlow Probability to 0.11/0.12

Thanks
------
- Remco de Boer and Stefan Pflüger for discussions on random number genration

1.1.0 (27.1.2020)
=================

This release switched to TensorFlow 2.0 eager mode. Please upgrade your TensorFlow installation if possible and change
your code (minimal changes) as described under "Behavioral changes".
In case this is currently impossible to do, please downgrade to < 1.1.0.

Major Features and Improvements
-------------------------------
 - full TF2 compatibility

Behavioral changes
------------------
 - `generate` now returns an eager Tensor. This is basically a numpy array wrapped by TensorFlow.
   To explicitly convert it to a numpy array, use the `numpy()` method of the eager Tensor.
 - `generate_tensor` is now depreceated, `generate` can directly be used instead.


Bug fixes and small changes
---------------------------

Requirement changes
-------------------
 - requires now TensorFlow >= 2.0.0


Thanks
------


1.0.4 (13-10-2019)
==========================


Major Features and Improvements
-------------------------------

Release to conda-forge, thanks to Chris Burr
