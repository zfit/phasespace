#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import os
import warnings

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

# PY23: remove try-except, keep try block
try:
    with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as requirements_file:
        requirements = requirements_file.read().splitlines()

    with open(os.path.join(here, 'requirements_dev.txt'), encoding='utf-8') as requirements_dev_file:
        requirements_dev = requirements_dev_file.read().splitlines()

    with open(os.path.join(here, 'README.rst'), encoding='utf-8') as readme_file:
        readme = readme_file.read()

    with open(os.path.join(here, 'HISTORY.rst'), encoding='utf-8') as history_file:
        history = history_file.read()
except TypeError:  # 'encoding' parameter not yet supported in python2

    with open(os.path.join(here, 'requirements.txt')) as requirements_file:
        requirements = requirements_file.read().splitlines()

    with open(os.path.join(here, 'requirements_dev.txt')) as requirements_dev_file:
        requirements_dev = requirements_dev_file.read().splitlines()

    with open(os.path.join(here, 'README.rst')) as readme_file:
        readme = readme_file.read()

    with open(os.path.join(here, 'HISTORY.rst')) as history_file:
        history = history_file.read()

    warnings.warn("Due to the end of lifetime, Python 2 support is not guaranteed and will stop "
                  "working in the future. It is HIGHLY recommended to use (the latest) Python 3 version.")

# split the developer requirements into setup and test requirements
if not requirements_dev.count("") == 1 or requirements_dev.index("") == 0:
    raise SyntaxError("requirements_dev.txt has the wrong format: setup and test "
                      "requirements have to be separated by one blank line.")
requirements_dev_split = requirements_dev.index("")

setup_requirements = requirements_dev[:requirements_dev_split]
test_requirements = requirements_dev[requirements_dev_split + 1:]  # +1: skip empty line

setup(
        author="Albert Puig Navarro",
        author_email='albert.puig@cern.ch',
        classifiers=[
            'Development Status :: 2 - Pre-Alpha',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: BSD License',
            'Natural Language :: English',
            "Programming Language :: Python :: 2",
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            ],
        description="Tensorflow implementation of the Raubold and Lynch method for n-body events",
        install_requires=requirements,
        license="BSD license",
        long_description=readme + '\n\n' + history,
        include_package_data=True,
        keywords='phasespace',
        name='phasespace',
        packages=find_packages(include=['phasespace']),
        setup_requires=setup_requirements,
        test_suite='tests',
        tests_require=test_requirements,
        url='https://github.com/zfit/phasespace',
        version='0.1.0',
        zip_safe=False,
        )
