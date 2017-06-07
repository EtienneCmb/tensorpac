"""
"""
import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='tensorpac',
    version='0.0',
    packages=find_packages(),
    description='Tensor-based Phase-Amplitude Coupling Python toolbox',
    long_description=read('README.md'),
    install_requires=[
        'numpy',
        'joblib',
    ],
    dependency_links=[],
    author='Etienne Combrisson',
    maintainer='Etienne Combrisson',
    author_email='e.combrisson@gmail.com',
    url='https://github.com/EtienneCmb/tensorpac',
    license=read('LICENSE'),
    include_package_data=True,
    keywords='phase-amplitude coupling PAC connectivity tensor neuroscience',
    classifiers=["Development Status :: 3 - Alpha",
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: Education',
                 'Intended Audience :: Developers',
                 'Topic :: Scientific/Engineering :: Visualization',
                 "Programming Language :: Python :: 3.5"
                 ])
