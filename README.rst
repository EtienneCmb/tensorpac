=========
Tensorpac
=========

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


.. figure::  https://github.com/EtienneCmb/tensorpac/blob/master/docs/source/picture/tp.png
   :align:   center

Description
-----------

Tensorpac is an Python open-source toolbox for computing Phase-Amplitude Coupling (PAC) using tensors and parallel computing for an efficient, and highly flexible modular implementation of PAC metrics both known and novel. Check out our `documentation <http://etiennecmb.github.io/tensorpac/>`_  for details.

Installation
------------

Tensorpac uses NumPy, SciPy and joblib for parallel computing. To get started, just open your terminal and run :


.. code-block:: console

    $ pip install tensorpac

Code snippet & illustration
---------------------------

.. code-block:: python

  from tensorpac import Pac
  from tensorpac.signals import pac_signals_tort

  # Dataset of signals artificially coupled between 10hz and 100hz :
  n_epochs = 20   # number of trials
  n_times = 4000  # number of time points
  sf = 512.       # sampling frequency

  # Create artificially coupled signals using Tort method :
  data, time = pac_signals_tort(f_pha=10, f_amp=100, noise=2, n_epochs=n_epochs,
                                dpha=10, damp=10, sf=sf, n_times=n_times)

  # Define a Pac object
  p = Pac(idpac=(6, 0, 0), f_pha='hres', f_amp='hres')
  # Filter the data and extract pac
  xpac = p.filterfit(sf, data)

  # plot your Phase-Amplitude Coupling :
  p.comodulogram(xpac.mean(-1), cmap='Spectral_r', plotas='contour', ncontours=5,
                 title=r'10hz phase$\Leftrightarrow$100Hz amplitude coupling',
                 fz_title=14, fz_labels=13)

  p.show()



.. figure::  https://github.com/EtienneCmb/tensorpac/blob/master/docs/source/picture/readme.png
   :align:   center
