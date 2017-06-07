.. _Methods:

Methods
=======

Modular philosophy
------------------

Digit description

Implemented methods
~~~~~~~~~~~~~~~~~~~

Pac methods
~~~~~~~~~~~

Methods used tensors :

* Mean Vector Length
* Kullback-Leibler
* Heigth-ratio
* ndPAC

Surrogates methods
~~~~~~~~~~~~~~~~~~

* No surrogates
* Swap phase/amplitude trials
* Swap amplitude time blocks
* Shuffle phase and amplitude time-series
* Shuffle phase time-series
* Shuffle amplitude time-series

.. todo::

    * Circular shifting
    * Time-lag

Normalization methods
~~~~~~~~~~~~~~~~~~~~~

* No normalization
* Substraction of the mean of surrogates
* Division of the mean of surrogates
* Substraction then division of the mean of surrogates
* Substraction by the mean, then division by the diviation of the surrogates (z-scored)


Link with publications
----------------------

Inputs
------

Main PAC class
~~~~~~~~~~~~~~

.. code-block:: python

    from tensorpac import Pac

Main PAC class
~~~~~~~~~~~~~~

.. autoclass:: tensorpac.pac.Pac

Filter the data
~~~~~~~~~~~~~~~

This method can be used to filter the data only in order to extract phase and amplitude.

.. automethod:: tensorpac.pac.Pac.filter

Compute PAC on filtered data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For those who already have filtered and computed the phase and amplitude, use this method to compute PAC.

.. automethod:: tensorpac.pac.Pac.fit

All in one method
~~~~~~~~~~~~~~~~~

Finally, use the following method to filter and compute the PAC directly.

.. automethod:: tensorpac.pac.Pac.filterfit

Plot PAC
~~~~~~~~

Small but useful method for plotting PAC in a comodulogram form. 

.. automethod:: tensorpac.pac.Pac.comodulogram