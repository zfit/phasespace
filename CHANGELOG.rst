*********
Changelog
*********

Develop
=======

This release switched to TensorFlow 2.0 eager mode. Please upgrade your TensorFlow installation if possible.
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




