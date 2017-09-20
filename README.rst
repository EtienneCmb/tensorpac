.. -*- mode: rst -*-

.. image:: https://travis-ci.org/EtienneCmb/tensorpac.svg?branch=master
    :target: https://travis-ci.org/EtienneCmb/tensorpac

.. image:: https://codecov.io/gh/EtienneCmb/tensorpac/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/EtienneCmb/tensorpac

.. image:: https://badge.fury.io/py/Tensorpac.svg
    :target: https://badge.fury.io/py/Tensorpac

Tensorpac
#########

.. figure::  https://github.com/EtienneCmb/tensorpac/blob/master/docs/source/picture/tp.png
   :align:   center

Description
===========

Tensorpac is an Python open-source toolbox for computing Phase-Amplitude Coupling (PAC) using tensors and parallel computing for an efficient, and highly flexible modular implementation of PAC metrics both known and novel. Check out our `documentation <http://etiennecmb.github.io/tensorpac/>`_  for details.

Installation
============

Tensorpac uses NumPy, SciPy and joblib for parallel computing. To get started, just open your terminal and run :

.. code-block:: shell

    pip install tensorpac

Code snippet & illustration
===========================

.. code-block:: python

  from tensorpac.utils import pac_signals_tort
  from tensorpac import Pac

  # Dataset of signals artificially coupled between 10hz and 100hz :
  n = 20     # number of datasets
  sf = 512.  # sampling frequency

  # Create artificially coupled signals using Tort method :
  data, time = pac_signals_tort(fpha=10, famp=100, noise=2, ntrials=n,
                                dpha=10, damp=10, sf=sf)

  # Define a PAC object :
  p = Pac(idpac=(4, 0, 0), fpha=(2, 20, 1, 1), famp=(60, 150, 5, 5),
          dcomplex='wavelet', width=12)
  # Filter the data and extract PAC :
  xpac = p.filterfit(sf, data, axis=1)

  # Plot your Phase-Amplitude Coupling :
  p.comodulogram(xpac.mean(-1), title='Contour plot with 5 regions',
                 cmap='Spectral_r', plotas='contour', ncontours=5)

  p.show()


.. figure::  https://github.com/EtienneCmb/tensorpac/blob/master/docs/source/picture/readme.png
   :align:   center

Contributors
============

* `Etienne Combrisson <http://etiennecmb.github.io>`_
* Juan L.P. Soto
* Timothy C. Nest
* `Karim Jerbi <www.karimjerbi.com>`_

