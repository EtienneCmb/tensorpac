.. _api:

API
===

.. contents::
   :local:
   :depth: 2

.. _fcncfc:

Compute phase-amplitude coupling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:mod:`tensorpac`:

.. currentmodule:: tensorpac

.. autosummary::
   :toctree: generated/

   Pac
   EventRelatedPac
   PreferredPhase

Utility functions
~~~~~~~~~~~~~~~~~

:py:mod:`tensorpac.utils`:

.. currentmodule:: tensorpac.utils

.. autosummary::
   :toctree: generated/

   PSD
   ITC
   BinAmplitude
   pac_vec
   pac_trivec

Generate synthetic signals
~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:mod:`tensorpac.signals`:

.. currentmodule:: tensorpac.signals

.. autosummary::
   :toctree: generated/

   pac_signals_wavelet
   pac_signals_tort

Statistics
~~~~~~~~~~

:py:mod:`tensorpac.stats`:

.. currentmodule:: tensorpac.stats

.. autosummary::
   :toctree: generated/

   test_stationarity


Individual methods
~~~~~~~~~~~~~~~~~~

:py:mod:`tensorpac.methods`:

PAC methods
+++++++++++

If you don't want to use the :class:`tensorpac.Pac` class, you can also manually import the method of your choice
and use it on phase / amplitude to compute PAC.

.. currentmodule:: tensorpac.methods

.. autosummary::
   :toctree: generated/

   mean_vector_length
   modulation_index
   heigths_ratio
   norm_direct_pac
   phase_locking_value
   gauss_cop_pac

Event Related PAC methods
+++++++++++++++++++++++++

.. currentmodule:: tensorpac.methods

.. autosummary::
   :toctree: generated/

   erpac
   ergcpac

Preferred phase
+++++++++++++++

.. currentmodule:: tensorpac.methods

.. autosummary::
   :toctree: generated/

   preferred_phase

Surrogates methods
++++++++++++++++++

.. currentmodule:: tensorpac.methods

.. autosummary::
   :toctree: generated/

   swap_pha_amp
   swap_blocks
   time_lag

Normalization
+++++++++++++

.. currentmodule:: tensorpac.methods

.. autosummary::
   :toctree: generated/

   normalize
