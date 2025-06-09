#!/usr/bin/env python
__usage__ = "setup.py command [--options]"
__description__ = "standard install script"
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

from setuptools import (setup, find_packages)
import glob

#-------------------------------------------------

# set up arguments
scripts = glob.glob('bin/w4t-*')

packages = find_packages()

# set up requirements
requires = [
    'numpy',
    'scipy',
    'matplotlib',
    'PyWavelets',
    'numpyro',
    'jax',
]

#------------------------

# install
setup(
    name = 'w4t',
    version = '0.0.0',
    url = 'https://github.com/reedessick/wavelets4turbulence',
    author = __author__,
    author_email = 'reed.essick@gmail.com',
    description = __description__,
    license = None,
    scripts = scripts,
    packages = packages,
    requires = requires,
)
