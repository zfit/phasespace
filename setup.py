#!/usr/bin/env python

"""The setup script."""

import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.rst"), encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open(os.path.join(here, "CHANGELOG.rst"), encoding="utf-8") as history_file:
    history = history_file.read()

with open(
    os.path.join(here, "requirements.txt"), encoding="utf-8"
) as requirements_file:
    requirements = requirements_file.read().splitlines()

with open(
    os.path.join(here, "requirements_dev.txt"), encoding="utf-8"
) as requirements_dev_file:
    requirements_dev = requirements_dev_file.read().splitlines()

tests_require = [
    "pytest",
    "pytest-xdist",
    "pytest-cov",
    "flaky",
    "coverage",
    "numpy",
    "matplotlib",
    "uproot",
    "uproot4",
    "scipy",
    "wget",
]
setup(
    long_description=readme.replace(":math:", "") + "\n\n" + history,
    install_requires=requirements,
    tests_require=tests_require,
    extras_require={"dev": requirements_dev + tests_require, "tests": tests_require},
    use_scm_version=True,
)
