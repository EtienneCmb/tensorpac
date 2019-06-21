.. _Installation:

Installation
============

Requirements
++++++++++++

Tensorpac relies on three packages :

* `NumPy <https://www.numpy.org/>`_
* `SciPy <https://www.scipy.org/>`_
* `Joblib <https://joblib.readthedocs.io/en/latest/>`_

Then if you want to be able to plot your results you'll need to install
`Matplotlib <https://matplotlib.org/>`_.

Standard installation
+++++++++++++++++++++

Tensorpac can be installed using pip. In a terminal, run the following command :

.. code-block:: shell

    pip install tensorpac

And if you want want to update to the latest version :

.. code-block:: shell

    pip install -U tensorpac

Install the most up-to-date version
+++++++++++++++++++++++++++++++++++

The latest version is hosted on `github <https://github.com/EtienneCmb/tensorpac>`_.
This is always going to be the most up-to-date version, with the latest features and fixes.
If you want to install this version, open a terminal and run the following commands :

.. code-block:: shell

    git clone https://github.com/EtienneCmb/tensorpac.git
    cd tensorpac/
    python setup.py develop

Finally, if you want to update your version you can use :

.. code-block:: shell

    git pull
