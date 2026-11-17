# Changelog

## 2.0.0 (17 Nov 2026)

### Behavioral changes

- NumPy is now the default backend. Select TensorFlow in Python with

  ```python
  import phasespace

  phasespace.set_backend("tensorflow")
  ```

  or through the environment with

  ```shell
  export PHASESPACE_BACKEND=tensorflow
  ```

### Requirement changes

- NumPy is the only required dependency. TensorFlow and TensorFlow Probability move to the `tf` extra, which the `fromdecay` extra includes. Install it to keep using the TensorFlow backend:

  ```shell
  pip install "phasespace[tf]"
  ```

- Python 3.14 and 3.15 are supported with the NumPy and JAX backends. The TensorFlow backend and the `fromdecay` extra remain limited to the Python versions that TensorFlow supports.

## 1.11.0 (15 Nov 2026)

### Major features and improvements

- NumPy and optional JAX backends are available alongside TensorFlow, with runtime backend selection through `phasespace.set_backend` and `phasespace.get_backend`. Existing decay objects can be reused after switching. TensorFlow remains the default backend in this release.

### Bug fixes and small changes

- Seeded resonance mass functions receive the generation random stream, zero Lorentz boosts no longer divide by zero, and NumPy mass checks accept equality.
- `GenMultiDecay` now has a readable representation that lists its decay modes and their probabilities.

### Requirement changes

- The `fromdecay` extra now requires `particle>=0.25.0` and `decaylanguage>=0.19.0`.
- Developer dependencies are now defined as dependency groups (`test`, `doc`, `dev`) instead of the `test`, `doc` and `dev` extras. Install them with `pip install -e .[all] --group dev` (pip 25.1 or newer) or `uv sync --extra all`.

### Documentation

- The documentation is now written in MyST Markdown and executed with MyST-NB. Read the Docs builds it with uv and creates previews for pull requests.

## 1.10.4 (16 Apr 2026)

### Requirement changes

- Python 3.10 to 3.13 are supported. Support for Python 3.9 is dropped.
- TensorFlow 2.19 or newer and TensorFlow Probability 0.25 or newer are required.

### Documentation

- The documentation uses the PyData Sphinx theme, which fixes its layout.

## 1.10.3 (23 Apr 2024)

### Bug fixes and small changes

- `vector` was imported unconditionally and was therefore a required dependency. It is optional again and available through the `vector` extra.

## 1.10.2 (16 Apr 2024)

Technical release that fixes the build and deployment. No changes to the package.

## 1.10.1 (16 Apr 2024)

Technical release that fixes the deployment. No changes to the package.

## 1.10.0 (16 Apr 2024)

### Major features and improvements

- [vector](https://vector.readthedocs.io/en/latest/index.html) is supported in `generate`: `boost_to` can be a Momentum Lorentz vector, and `as_vectors=True` returns the boosted particles as vectors.

### Requirement changes

- Python 3.12 is supported. Support for Python 3.8 is dropped.
- TensorFlow 2.16 or newer is required.

## 1.9.0 (20 Jul 2023)

### Requirement changes

- Python 3.11 is supported. Support for Python 3.7 is dropped.

## 1.8.0 (27 Jan 2023)

### Requirement changes

- zfit 0.10.0 or newer and zfit-physics 0.3.0 or newer are required.
- uproot and awkward are pinned to ~4 and ~1, respectively.

## 1.7.0 (1 Sep 2022)

The `tf` and `tensorflow` extras are added to the requirements. If you intend to use phasespace with TensorFlow in the future (and not only NumPy), always install with `phasespace[tf]`.

### Requirement changes

- TensorFlow 2.7 or newer is required.
- Python 3.7 to 3.10 are supported.

## 1.6.0 (14 Apr 2022)

### Major features and improvements

- `GenMultiDecay` gives better control over the decay mass of non-stable particles.
- `GenMultiDecay` has a new `particle_model_map` argument. This is a dict where the key is a particle name and the value is a mass function name. The feature is shown in the [GenMultiDecay Tutorial](https://phasespace.readthedocs.io/en/stable/fromdecay.html).

## 1.5.0 (27 Nov 2021)

### Major features and improvements

- Decays can be generated from a DecayChain of the [decaylanguage](https://github.com/scikit-hep/decaylanguage) package from Scikit-HEP. This is in the new subpackage "fromdecay" and can be used by installing the extra with `pip install phasespace[fromdecay]`.

### Requirement changes

- Support for Python 3.6 is dropped.

### Thanks

- To Simon Thor for contributing the `fromdecay` subpackage.

## 1.4.2 (5 Nov 2021)

### Requirement changes

- The restriction on TensorFlow is loosened to allow version 2.7 (and 2.5, 2.6).

## 1.4.1 (27 Aug 2021)

### Requirement changes

- The restriction on TensorFlow is loosened to allow version 2.6 (and 2.5).

## 1.4.0 (11 Jun 2021)

### Requirement changes

- TensorFlow 2.5 is required, as 2.4 breaks some functionality.

## 1.3.0 (28 May 2021)

### Major features and improvements

- Python 3.9 is supported.
- TensorFlow 2.5 is supported.
- Compilation in `tf.function`s is improved, with XLA used where applicable.
- For developers: setup, CI and more are modernized.

### Thanks

- Remco de Boer for many commits and cleanups.

## 1.2.0 (17 Dec 2020)

### Major features and improvements

- Python 3.8 is supported.
- Eager execution can be enabled with `tf.config.run_functions_eagerly(True)` or the environment variable `PHASESPACE_EAGER`.
- Random number generation is deterministic via a seed or a `tf.random.Generator` instance.

### Requirement changes

- TensorFlow is restricted to 2.3/2.4.
- TensorFlow Probability is restricted to 0.11/0.12.

### Thanks

- Remco de Boer and Stefan Pflüger for discussions on random number generation.

## 1.1.1 (17 Feb 2020)

Technical release that switches to manual versioning. No changes to the package.

## 1.1.0 (27 Jan 2020)

This release switched to TensorFlow 2.0 eager mode. Please upgrade your TensorFlow installation if possible and change your code (minimal changes) as described under "Behavioral changes". In case this is currently impossible to do, please downgrade to < 1.1.0.

### Major features and improvements

- Full TensorFlow 2 compatibility.

### Behavioral changes

- `generate` now returns an eager Tensor. This is basically a numpy array wrapped by TensorFlow. To explicitly convert it to a numpy array, use the `numpy()` method of the eager Tensor.
- `generate_tensor` is now deprecated, `generate` can directly be used instead.

### Requirement changes

- TensorFlow 2.0.0 or newer is required.

## 1.0.4 (13 Oct 2019)

Release to conda-forge, thanks to Chris Burr.

## 1.0.3 (13 Oct 2019)

Release to conda-forge, thanks to Chris Burr. This release cannot be installed with pip; use 1.0.4 instead.

## 1.0.2 (3 Jun 2019)

The JOSS paper is added and a problem with Zenodo is fixed.

## 1.0.1 (20 May 2019)

First stable release of phasespace.
