#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: 3-clause BSD
import os
from setuptools import setup, find_packages

__version__ = "0.5.5"
NAME = 'Tensorpac'
AUTHOR = "Etienne Combrisson"
MAINTAINER = "Etienne Combrisson"
EMAIL = 'e.combrisson@gmail.com'
KEYWORDS = "phase-amplitude-coupling pac tensor"
DESCRIPTION = "Tensor-based Phase-Amplitude Coupling"
URL = 'http://etiennecmb.github.io/tensorpac/'
DOWNLOAD_URL = ("https://github.com/EtienneCmb/tensorpac/archive/v" +
                __version__ + ".tar.gz")
# Data path :
PACKAGE_DATA = {}


def read(fname):
    """Read README and LICENSE."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name=NAME,
    version=__version__,
    packages=find_packages(),
    package_dir={'tensorpac': 'tensorpac'},
    package_data=PACKAGE_DATA,
    include_package_data=True,
    description=DESCRIPTION,
    long_description=read('README.rst'),
    platforms='any',
    setup_requires=['numpy', 'joblib'],
    install_requires=[
        "numpy>=1.12",
        "scipy",
        "joblib"
    ],
    dependency_links=[],
    author=AUTHOR,
    maintainer=MAINTAINER,
    author_email=EMAIL,
    url=URL,
    download_url=DOWNLOAD_URL,
    license=read('LICENSE'),
    keywords=KEYWORDS,
    classifiers=["Development Status :: 3 - Alpha",
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: Education',
                 'Intended Audience :: Developers',
                 'Topic :: Scientific/Engineering :: Visualization',
                 "Programming Language :: Python :: 3.5"
                 ])
