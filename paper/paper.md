---
title: 'phasespace: $n$-body phase space generation in Python'
tags:
  - high energy physics
  - tensorflow
  - relativistic kinematics
authors:
  - name: Albert Puig Navarro
    orcid: 0000-0001-8868-2947
    affiliation: "1"
  - name: Jonas Eschle
    orcid: 0000-0002-7312-3699
    affiliation: "1"
affiliations:
 - name: Physik-Institut, Universität Zürich, Zürich (Switzerland)
   index: 1
date: 3 June 2019
bibliography: paper.bib
---

# Summary

The use of simulated particle decays is very common in experimental particle physics.
They are used to study a wide variety of aspects of a physics analysis, such as signal response, detector effects, and the efficiency of selection requirements, in a controlled manner.
While it is possible to encode complex physics dynamics in these simulations (at the cost of increased complexity and larger computer requirements), in many cases it is enough to generate these simulated samples as if no physics occurred, i.e., in an isotropic way.
This type of generation, called "phase space generation" is very fast and offers simple and predictable patterns, making it very attractive as a first step of many physics analyses.

The ``phasespace`` package implements phase space event generation based on the Raubold and Lynch method described in Ref. [@James:1968gu].
This method was previously implemented in the ``GENBOD`` function of the FORTRAN-based ``CERNLIB`` library, and posteriorly ported to C++ in the context of the ROOT toolkit [@Brun:1997pa] as the ``TGenPhaseSpace`` class, which is currently the most used in particle physics.
The ``phasespace`` package provides a pure-Python implementation of the Raubold and Lynch method implemented on the ``Tensorflow`` platform [@tensorflow2015-whitepaper].
Unlike ``TGenPhaseSpace``, the ``phasespace`` approach offers perfect integration with the scientific Python ecosystem (*numpy*, *pandas*, *scikit-learn*...) while at the same time offering excellent performance both in CPU and GPU thanks to its ``Tensorflow`` foundation.

In addition, ``phasespace`` allows the generation of complex multi-decay chains, including non-constant masses for some of the involved particles (needed for the simulation of resonant particles).
This functionality opens the door for its use as the basis for importance sampling in Dalitz and amplitude decay fitters, which typically need to implement their own solution based on ``TGenPhaseSpace``;
in this sense, ``phasespace`` is currently being used for the implementation of amplitude fit integrals in the ``zfit`` fitter [@zfit].

The correctness of ``phasespace`` has been validated (and is continuously validated through its test suite) against ``TGenPhaseSpace`` and the ``RapidSim`` package [@Cowan:2016tnm], an application for the simulation of heavy-quark hadron decays;
this latter application also uses ``TGenPhaseSpace``, but adds features such as multi-decay chains and simulation of the kinematics found in colliders such as the LHC.

In summary, ``phasespace`` was designed to fill an important gap in the recent paradigm shift of particle physics analysis towards integration with the scientific Python ecosystem, and to do so also adding more advanced functionality than its C++-based predecessors.
Its ease of use, clear interface and easy interoperability with other packages provides a solid foundation to build upon in the quest for a full Python-based particle physics analysis software stack.
The source code for ``phasespace`` has been archived to Zenodo with the linked DOI: [@zenodo]

# Acknowledgements

A.P. acknowledges support from the Swiss National Science Foundation under contract 168169.

# References
