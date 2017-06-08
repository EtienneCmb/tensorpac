Tensorpac
#########

Tensorpac is an Python open-source toolbox for computing Phase-Amplitude Coupling (PAC) using tensors and parallel computing. On top of that, we designed a modular implementation with a relatively large amount of parameters.
We provide a set of `examples <https://github.com/EtienneCmb/tensorpac/tree/master/examples>`_.

This package was developped in collaboration with **Juan L.P. Soto**

.. figure::  picture/tp.png
   :align:   center

Installation:
*************

Tensorpac is only based on NumPy and use `Joblib <https://pythonhosted.org/joblib/>`_ for parallel computing. For the installation, in a terminal run :

.. code-block:: bash

    git clone https://github.com/EtienneCmb/tensorpac.git tensorpac
    cd tensorpac
    pip setup.py install

What's new?
***********

* New in version v0.3
    
    * New doc

Todo list
*********

.. todo::

    * Generalized Linear Model (GLM - Lakata, 2005)
    * Event Related PAC (ERPAC - Voytek, 2013)
    * Memory tracking
    * Example Tensorpac Vs pacpy
    * Installation using *pip install tensorpac*

Contents:
*********

.. toctree::
   :maxdepth: 3

   tutorial
   methods
   pyfcn
   utils
   visu


Indices and tables
##################

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

