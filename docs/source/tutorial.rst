.. _Tutorial:

Implemented methods
===================

Beginning with raw data, the first step to computing PAC is to extract the phase and the amplitude across a range of frequency bands. There are two ways to do this :

* Filter the data in the frequency band, then use the hilbert transform to pass from real to complex space. 
* Convolve your data with a wavelet with a width, corresponding to the desired frequency band.

Next we will have to extract the phase and the amplitude from the resultant decomposition. 

Once we have resolved both phase and amplitude signals, we shall derive our PAC value. This is done in three steps :

* **Compute the PAC :**  From the two time series of phase and amplitude, this will give us a single coupling value
* **Compute surogates :** The PAC measure is quite sensitive to noise present in the data. In addition, filtering may augment artefacts. To minimize these effects, we compute surrogates. Surrogates are obtained by altering slightly either the phase or the amplitude and re-computing the PAC. This procedure is then repeated over 50, 100, 200 or 1000 permutations to produce a reliable distribution.
* **Correct the PAC measurement :** Once we have the PAC and the distribution of surrogates, we can subtract the mean of the surrogates. By taking the mean of the distribution, we limit our results to those strongest, and mitigate the impact of noise and artefact.

The many existing PAC implemenations nearly always propose some variation on these three steps. How can we be sure, though, that any particular implemenation will outperform any other? 

Modular philosophy
------------------

To answer to this question, we've developed a modular implementation allowing users to combine methods from existing PAC implementations in novel ways. Until recently, it has been difficult to compare existing and novel combinations of PAC implementations, due to long computation times. By leveraging tensor and parallel computing, however, Tensorpac offers new possibilities for determining the optimal PAC implementation for a given data set.

Tensorpac makes combining methods as simple as defining the **idpac** tuple (or list/array) of three digits, each referring to one of the implemented methods. 

Pac methods
~~~~~~~~~~~

The first digit defines the PAC method :

* 1 - Mean Vector Length (MVL - Canolty, 2006)
* 2 - Kullback-Leibler Divergence (KLD - Tort, 2010)
* 3 - Heigth-ratio (HR - Lakatos, 2005)
* 4 - Normalized Direct PAC (ndPAC - Ozkürt, 2012)
* 5 - Phase Synchrony (PS - Lakata, 2005)
* 6 - Generalized Linear Model (GLM - Lakata, 2005) [IN PROGRESS]
* 7 - Event Related PAC (ERPAC - Voytek, 2013) [IN PROGRESS]

[`PAC methods script <https://github.com/EtienneCmb/tensorpac/blob/master/examples/4_ComparePacMethods.py>`_]

.. figure::  picture/4_pacmeth.png
   :align:   center

   `PAC methods comparison <https://github.com/EtienneCmb/tensorpac/blob/master/docs/source/picture/4_pacmeth.png>`_.

Surrogates methods
~~~~~~~~~~~~~~~~~~

The second digit defines the method for surrogates computation :

* 0 - No surrogates
* 1 - Swap phase/amplitude trials (Tort, 2010)
* 2 - Swap amplitude time blocks (Bahramisharif, 2013)
* 3 - Shuffle phase and amplitude time-series
* 4 - Shuffle phase time-series
* 5 - Shuffle amplitude time-series
* 6 - Time-lag (Canolty, 2006)

[`Surrogate methods script <https://github.com/EtienneCmb/tensorpac/blob/master/examples/5_CompareSurrogatesMethods.py>`_]

.. figure::  picture/5_surrometh.png
   :align:   center

   `Surrogate methods comparison <https://github.com/EtienneCmb/tensorpac/blob/master/docs/source/picture/5_surrometh.png>`_.

Normalization methods
~~~~~~~~~~~~~~~~~~~~~

The third digit is defines the normalization (or correction) procedure :

* 0 - No normalization
* 1 - Substraction of the mean of surrogates
* 2 - Division of the mean of surrogates
* 3 - Substraction then division of the mean of surrogates
* 4 - Substraction by the mean, then division by the diviation of the surrogates (z-scored)

[`Normalization methods script <https://github.com/EtienneCmb/tensorpac/blob/master/examples/6_CompareNormalizationMethods.py>`_]

.. figure::  picture/6_normmeth.png
   :align:   center

   `Normalization methods comparison <https://github.com/EtienneCmb/tensorpac/blob/master/docs/source/picture/6_normmeth>`_.

Example
~~~~~~~

* idpac = (2, 0, 0) => KLD + No surrogates + No normalization
* idpac = (1, 3, 3) => MVL + Shuffle phase and amplitude time-series + Substraction then division of the mean of surrogates
* idpac = (5, 2, 1) => PS + Swap amplitude time blocks + Substraction of the mean of surrogates


.. Link with publications
.. ----------------------

.. * Canolty, 2006 : idpac = ()
.. * Tort, 2010 : idpac = ()
