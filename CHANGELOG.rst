*********
Changelog
*********

Develop
=======


Major Features and Improvements
-------------------------------

Behavioral changes
------------------


Bug fixes and small changes
---------------------------
- improved compilation in tf.functions, use of XLA where applicable

Requirement changes
-------------------


Thanks
------

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
