#!/usr/bin/env python

"""The setup script."""

import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.rst"), encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open(os.path.join(here, "CHANGELOG.rst"), encoding="utf-8") as history_file:
    history = history_file.read()

setup(
    long_description=readme.replace(":math:", "") + "\n\n" + history,
    use_scm_version=True,
)
