# PhaseSpace

[![JOSS](https://joss.theoj.org/papers/10.21105/joss.01570/status.svg)](https://doi.org/10.21105/joss.01570)
[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2591993.svg)](https://doi.org/10.5281/zenodo.2591993)
[![PyPI status](https://img.shields.io/pypi/status/phasespace.svg)](https://pypi.org/project/phasespace/)
[![Python versions](https://img.shields.io/pypi/pyversions/phasespace.svg)](https://pypi.org/project/phasespace/)
[![Tests](https://github.com/zfit/phasespace/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/zfit/phasespace/actions/workflows/ci.yml?query=branch%3Amain)
[![Codecov](https://codecov.io/gh/zfit/phasespace/branch/main/graph/badge.svg)](https://codecov.io/gh/zfit/phasespace)
[![Documentation Status](https://readthedocs.org/projects/phasespace/badge/?version=stable)](https://phasespace.readthedocs.io/en/stable)
[![Gitter chat](https://badges.gitter.im/zfit/phasespace.svg)](https://app.gitter.im/#/room/#zfit_phasespace:gitter.im)
[![PyPI version](https://badge.fury.io/py/phasespace.svg)](https://badge.fury.io/py/phasespace)

Python implementation of the Raubold and Lynch method for *n*-body events with support for multiple computational backends (TensorFlow, NumPy, JAX).

The code is based on the GENBOD function (W515 from CERNLIB), documented in [^1] and tries to follow it as closely as possible.

Detailed documentation, including the API, can be found in <https://phasespace.readthedocs.io>. Don't hesitate to join our [gitter](https://app.gitter.im/#/room/#zfit_phasespace:gitter.im) channel for questions and comments.

If you use phasespace in a scientific publication we would appreciate citations to the [JOSS](https://joss.theoj.org/papers/10.21105/joss.01570) publication:

```bibtex
@article{puig_eschle_phasespace-2019,
  title = {phasespace: n-body phase space generation in Python},
  doi = {10.21105/joss.01570},
  url = {https://doi.org/10.21105/joss.01570},
  year = {2019},
  month = {oct},
  publisher = {The Open Journal},
  author = {Albert Puig and Jonas Eschle},
  journal = {Journal of Open Source Software}
}
```

Free software: BSD-3-Clause.

[^1]: F. James, [Monte-Carlo phase space](https://doi.org/10.5170/CERN-1968-015), CERN Yellow Report CERN-68-15 (1968)

## Why?
Lately, data analysis in High Energy Physics (HEP), traditionally performed within the [ROOT](https://root.cern.ch) ecosystem, has been moving more and more towards Python. The possibility of carrying out purely Python-based analyses has become real thanks to the development of many open source Python packages, which have allowed to replace most ROOT functionality with Python-based packages.

One of the aspects where this is still not possible is in the random generation of *n*-body phase space events, which are widely used in the field, for example to study kinematics of the particle decays of interest, or to perform importance sampling in the case of complex amplitude models. This has been traditionally done with the [TGenPhaseSpace](https://root.cern.ch/doc/master/classTGenPhaseSpace.html) class, which is based of the GENBOD function of the CERNLIB FORTRAN libraries and which requires a full working ROOT installation.

This package aims to address this issue by providing a TensorFlow-based implementation of such a function to generate *n*-body decays without requiring a ROOT installation. Additionally, an oft-needed functionality to generate complex decay chains, not included in `TGenPhaseSpace`, is also offered, leaving room for decaying resonances (which don't have a fixed mass, but can be seen as a broad peak).

## Installing

To install with pip:

```shell
pip install phasespace
```

This is the preferred method to install `phasespace`, as it will always install the most recent stable release. To install the necessary dependencies to be used with [DecayLanguage](https://github.com/scikit-hep/decaylanguage), use

```shell
pip install "phasespace[fromdecay]"
```

TensorFlow remains installed by default. For JAX support, install the optional extra:

```shell
pip install "phasespace[jax]"
```

## Backend Selection

Phasespace supports **TensorFlow** (default), **NumPy**, and **JAX**. The backend can be selected at runtime using the public API:

```python
import phasespace

decay = phasespace.nbody_decay(5279, [139, 493])
phasespace.set_backend("numpy")
numpy_weights, numpy_particles = decay.generate(n_events=1000, seed=42)

phasespace.set_backend("jax")
jax_weights, jax_particles = decay.generate(n_events=1000, seed=42)
print(phasespace.get_backend())  # BackendType.JAX
```

Alternatively, the initial backend can also be selected through an environment variable:

```shell
export PHASESPACE_BACKEND=numpy
# Other choices: tensorflow, jax
```

The backend applies to the whole program, so avoid switching it while events are being generated. Arrays you already generated keep their original type. A random generator belongs to the backend it was created with, so after switching, pass an integer seed or create a new generator. The same seed gives the same events within one backend, but different backends produce different events.

Custom mass functions should use operations that work with the selected backend. The mass distributions in `phasespace.fromdecay` are built on zfit, so use the TensorFlow backend with them.

## How to use

Phasespace can directly be used to generate from a DecayChain using the [DecayLanguage](https://github.com/scikit-hep/decaylanguage) package as [explained in the tutorial](https://phasespace.readthedocs.io/en/stable/fromdecay.html).

The generation of simple *n*-body decays can be done using the `nbody_decay` shortcut to create a decay chain with a very simple interface: one needs to pass the mass of the top particle and the masses of the children particle as a list, optionally giving the names of the particles. Then, the `generate` method can be used to produce the desired sample. For example, to generate $B^0\to K\pi$, we would do:

```python
import phasespace

B0_MASS = 5279.65
PION_MASS = 139.57018
KAON_MASS = 493.677

weights, particles = phasespace.nbody_decay(B0_MASS, [PION_MASS, KAON_MASS]).generate(n_events=1000)
```

Behind the scenes, this function runs the TensorFlow graph. It returns `tf.Tensor`, which, as TensorFlow 2.x is in eager mode, is basically a numpy array. Any `tf.Tensor` can be explicitly converted to a numpy array by calling `tf.Tensor.numpy()` on it. The `generate` function returns a `tf.Tensor` of 1000 elements in the case of `weights` and a list of `n particles` (2) arrays of (1000, 4) shape, where each of the 4-dimensions corresponds to one of the components of the generated Lorentz 4-vector. All particles are generated in the rest frame of the top particle; boosting to a certain momentum (or list of momenta) can be achieved by passing the momenta to the `boost_to` argument.

Sequential decays can be handled with the `GenParticle` class (used internally by `generate`) and its `set_children` method. As an example, to build the $B^{0}\to K^{*}\gamma$ decay in which $K^*\to K\pi$, we would write:

```python
from phasespace import GenParticle

B0_MASS = 5279.65
KSTARZ_MASS = 895.55
PION_MASS = 139.57018
KAON_MASS = 493.677

kaon = GenParticle('K+', KAON_MASS)
pion = GenParticle('pi-', PION_MASS)
kstar = GenParticle('K*', KSTARZ_MASS).set_children(kaon, pion)
gamma = GenParticle('gamma', 0)
bz = GenParticle('B0', B0_MASS).set_children(kstar, gamma)

weights, particles = bz.generate(n_events=1000)
```

Where we have used the fact that `set_children` returns the parent particle. In this case, `particles` is a `dict` with the particle names as keys:

```python
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
 'pi-': array([[  982.71247896, -1085.66304109,   726.16598675,  1640.48313937],
       [ -134.0528036 ,   -49.3079351 ,   -25.76067389,   201.35757766],
       [  309.50867032,  -999.97579937,  -804.0709006 ,  1327.31196961],
      ...,
       [  582.98445936,  -383.85197629,   445.36862511,   839.6703323 ],
       [ -305.02604662,  1361.12915468,  1029.29248526,  1739.14815935],
       [  479.39757002,  -268.78145413,  1397.86577911,  1508.50060384]])}
```

The `GenParticle` class is able to cache the graphs so it is possible to generate in a loop without overhead:

```python
for i in range(10):
    weights, particles = bz.generate(n_events=1000)
    ...
    # (do something with weights and particles)
    ...
```

This way of generating is recommended in the case of large samples, as it allows to benefit from parallelisation while at the same time keep the memory usage low.

If we want to operate with the TensorFlow graph instead, we can use the `generate_tensor` method of `GenParticle`, which has the same signature as `generate`.

More examples can be found in the `tests` folder and in the [documentation](https://phasespace.readthedocs.io/en/stable/usage.html).

## Physics validation

Physics validation is performed continuously in the included tests (`tests/test_physics.py`), run through GitHub Actions. This validation is performed at two levels:

- In simple *n*-body decays, the results of `phasespace` are checked against `TGenPhaseSpace`.
- For sequential decays, the results of `phasespace` are checked against [RapidSim](https://github.com/gcowan/RapidSim/), a "fast Monte Carlo generator for simulation of heavy-quark hadron decays". In the case of resonances, differences are expected because our tests don't include proper modelling of their mass shape, as it would require the introduction of further dependencies. However, the results of the comparison can be expected visually.

The results of all physics validation performed by the `tests_physics.py` test are written in `tests/plots`.

## Contributing

Contributions are always welcome, please have a look at the [Contributing guide](https://phasespace.readthedocs.io/en/stable/contributing.html).
