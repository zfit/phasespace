#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.rst'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open(os.path.join(here, 'CHANGELOG.rst'), encoding='utf-8') as history_file:
    history = history_file.read()

with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as requirements_file:
    requirements = requirements_file.read().splitlines()

with open(os.path.join(here, 'requirements_dev.txt'), encoding='utf-8') as requirements_dev_file:
    requirements_dev = requirements_dev_file.read().splitlines()

# split the developer requirements into setup and test requirements
if requirements_dev.count("") != 1 or requirements_dev.index("") == 0:
    raise SyntaxError("requirements_dev.txt has the wrong format: setup and test "
                      "requirements have to be separated by one blank line.")
requirements_dev_split = requirements_dev.index("")

test_requirements = requirements_dev[requirements_dev_split + 1:]  # +1: skip empty line

setup(
    long_description=readme.replace(":math:", "") + '\n\n' + history,
    install_requires=requirements,
    tests_require=test_requirements,
    extras_require={
        'dev': requirements_dev
    },
    use_scm_version=True,
)
