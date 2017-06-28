.. -*- mode: rst -*-

.. image:: https://travis-ci.org/EtienneCmb/tensorpac.svg?branch=master
    :target: https://travis-ci.org/EtienneCmb/tensorpac

.. image:: https://codecov.io/gh/EtienneCmb/tensorpac/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/EtienneCmb/tensorpac

.. image:: https://badge.fury.io/py/Tensorpac.svg
    :target: https://badge.fury.io/py/Tensorpac

Tensorpac
#########

Tensorpac is an Python open-source toolbox for computing Phase-Amplitude Coupling (PAC) using tensors and parallel computing. On top of that, we designed a modular implementation with a relatively large amount of parameters.
We provide a set of `examples <https://github.com/EtienneCmb/tensorpac/tree/master/examples>`_.

.. figure::  picture/tp.png
   :align:   center

Installation:
*************

Tensorpac is based on NumPy, SciPy and use `Joblib <https://pythonhosted.org/joblib/>`_ for parallel computing. For the installation, in a terminal run :

.. code-block:: bash

    pip install tensorpac

What's new?
***********

* New in version v0.5.2
    
    * :ref:`erpac` (Voytek et al. 2013)
    * pip installation

* New in version v0.5.1
    
    * Compute and plot :ref:`pp`
    * Bug fixing

Todo list
*********

.. todo::

    * Generalized Linear Model (GLM - Lakatos, 2005)
    * PAC using PSD
    * Morse's wavelets

Contents:
*********

.. toctree::
   :maxdepth: 3

   methods
   pyfcn
   utils
   stats
   visu


Indices and tables
##################

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

