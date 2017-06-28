.. _Python functions:

Python functions
=================

The first thing to do is to define a `pacobj`. Then, from this objects, you can use one of the following methods :

* :ref:`filtdata` : use this method to extract the phase and the amplitude.
* :ref:`pacfilt` : if you already extracted your phase and amplitude, use this method to compute pac directly on it. 
* :ref:`filtpac` : this is the all in one method. Starting from fresh data, this method will extract phase and amplitude and return the PAC estimation.
* :ref:`pp` : compute the preferred-phase.
* :ref:`erpac` : compute the Event-Related Phase-Amplitude Coupling (Voytek et al. 2013).

.. _pacobj:

PAC object
~~~~~~~~~~

.. code-block:: python

    from tensorpac import Pac

.. autoclass:: tensorpac.pac.Pac

.. _filtdata:

Filter the data
~~~~~~~~~~~~~~~

This method can be used to filter the data only in order to extract phase and amplitude.

.. automethod:: tensorpac.pac.Pac.filter

.. _pacfilt:

Compute PAC on filtered data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For those who already have filtered and computed the phase and amplitude, use this method to compute PAC.

.. automethod:: tensorpac.pac.Pac.fit

.. _filtpac:

Filter then compute PAC
~~~~~~~~~~~~~~~~~~~~~~~

Use the following method to filter and compute the PAC directly.

.. automethod:: tensorpac.pac.Pac.filterfit

.. _pp:

Preferred-phase
~~~~~~~~~~~~~~~

Compute the preferred-phase (PP, see `this PP example <https://github.com/EtienneCmb/tensorpac/tree/master/examples/13_PreferredPhase.py>`_). 

.. figure::  picture/PP_planningVsExec.png
   :align:   center

   Example of preferred-phase for a motor task. This show how the amplitude is distributed according to phase values during a motor planning phase and a motor execution phase.  

.. automethod:: tensorpac.pac.Pac.pp

.. _erpac:

Event-Related Phase-Amplitude Coupling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute the ERPAC (see `this ERPAC example <https://github.com/EtienneCmb/tensorpac/tree/master/examples/14_EventRelatedPAC.py>`_). 

.. figure::  picture/erpac.png
   :align:   center

   Example of Event-Related Phase-Amplitude Coupling (ERPAC, Voytek et al. 2013)

.. automethod:: tensorpac.pac.Pac.erpac
