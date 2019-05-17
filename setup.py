#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as requirements_file:
    requirements = requirements_file.read().splitlines()

with open(os.path.join(here, 'requirements_dev.txt'), encoding='utf-8') as requirements_dev_file:
    requirements_dev = requirements_dev_file.read().splitlines()

with open(os.path.join(here, 'README.rst'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open(os.path.join(here, 'HISTORY.rst'), encoding='utf-8') as history_file:
    history = history_file.read()

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
    maintainer="zfit",
    maintainer_email='zfit@physik.uzh.ch',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    description="TensorFlow implementation of the Raubold and Lynch method for n-body events",
    install_requires=requirements,
    license="BSD license",
    long_description=readme.replace(":math:", "") + '\n\n' + history,
    include_package_data=True,
    keywords='phasespace',
    name='phasespace',
    packages=find_packages(include=['phasespace']),
    setup_requires=setup_requirements,
    python_requires=">=3.6",
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/zfit/phasespace',
    version='1.0.0',
    zip_safe=False,
)
