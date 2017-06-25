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

Tensorpac is an Python open-source toolbox for computing Phase-Amplitude Coupling (PAC) using tensors and parallel computing. On top of that, we designed a modular implementation with a relatively large amount of parameters. Checkout the `documentation <http://etiennecmb.github.io/tensorpac/>`_  for further details.

Installation
============

Tensorpac use NumPy, SciPy and joblib for parallel computing. In a terminal, run :

.. code-block:: shell

    pip install tensorpac

Code snippet & illustration
===========================

.. code-block:: python

    import matplotlib.pyplot as plt
    from tensorpac.utils import PacSignals
    from tensorpac import Pac

    # Dataset of signals artificially coupled between 10hz and 100hz :
    n = 100  # number of datasets
    data, time = PacSignals(fpha=10, famp=100, noise=3, ndatasets=n, dpha=10, damp=10)

    # Extract PAC :
    p = Pac(idpac=(4, 0, 0), fpha=(2, 30, 1, 1), famp=(60, 150, 5, 5),
            dcomplex='wavelet', width=12)
    xpac, pval = p.filterfit(1024, data, data, axis=1, nperm=100)

    # Plot your Phase-Amplitude Coupling :
    p.comodulogram(xpac.mean(-1), title='Contour plot with 5 regions',
                   cmap='Spectral_r', plotas='contour', ncontours=5, vmin=60, vmax=300)

    plt.show()


.. figure::  https://github.com/EtienneCmb/tensorpac/blob/master/docs/source/picture/readme.png
   :align:   center

Contributors
============

* `Etienne Combrisson <http://etiennecmb.github.io>`_
* Juan L.P. Soto
* `Karim Jerbi <www.karimjerbi.com>`_

