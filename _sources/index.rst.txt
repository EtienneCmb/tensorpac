.. -*- mode: rst -*-

.. raw:: html

  <br>


.. image:: https://github.com/EtienneCmb/tensorpac/workflows/Tensorpac/badge.svg
    :target: https://github.com/EtienneCmb/tensorpac/workflows/Tensorpac

.. image:: https://travis-ci.org/EtienneCmb/tensorpac.svg?branch=master
    :target: https://travis-ci.org/EtienneCmb/tensorpac

.. image:: https://circleci.com/gh/EtienneCmb/tensorpac/tree/master.svg?style=svg
    :target: https://circleci.com/gh/EtienneCmb/tensorpac/tree/master

.. image:: https://ci.appveyor.com/api/projects/status/0arxtw05583gc3e2/branch/master?svg=true
    :target: https://ci.appveyor.com/project/EtienneCmb/tensorpac/branch/master

.. image:: https://codecov.io/gh/EtienneCmb/tensorpac/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/EtienneCmb/tensorpac

.. image:: https://badge.fury.io/py/tensorpac.svg
    :target: https://badge.fury.io/py/tensorpac

.. image:: https://pepy.tech/badge/tensorpac
    :target: https://pepy.tech/project/tensorpac

.. image:: https://badges.gitter.im/EtienneCmb/tensorpac.svg
    :target: https://gitter.im/EtienneCmb/tensorpac?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge

Tensorpac
#########

Tensorpac is an open-source Python toolbox for computing Phase-Amplitude Coupling (PAC) using tensors and parallel computing. This software provides a modular implementation which allows one to combine existing methods for measuring PAC and chance distribution.

.. figure::  picture/tp.png
   :align:   center


Tensor based implementation
***************************

In general, most of the softwares implemented the PAC in a vectorial fashion. This means that the PAC is computed between a single vector of phase and a single vector of amplitude. One of the limitation of this approach is that it can be relatively slow when exploring multi-dimensional data (e.g number of electrodes / sensors, number of trials, several frequency bands etc.). Tensorpac uses a different approach, using the Einstein summation, where the PAC is implemented in order to support multi-dimensional arrays (i.e tensors). This type of implementation can drastically decrease computational cost, especially if it's combined with parallel computing as it is the case in Tensorpac.

.. figure::  picture/10_detailed_loop_vs_tensor.png
   :align:   center

   On the left, a traditional loop implementation to compute PAC between vectors. On the right, an illustration of the tensor-based implementation.


Contents:
*********

.. toctree::
   :maxdepth: 2

   install
   tutorial
   api
   auto_examples/index.rst
   contributors
   cite
   community
