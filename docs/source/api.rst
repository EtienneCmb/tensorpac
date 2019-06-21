.. _api:

API
===

.. contents::
   :local:
   :depth: 2

.. _fcncfc:

Compute phase-amplitude coupling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: tensorpac

.. autosummary::
   :toctree: generated/

   Pac

Generate synthetic signals
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: tensorpac

.. autosummary::
   :toctree: generated/

   pac_signals_wavelet
   pac_signals_tort


Individual methods
~~~~~~~~~~~~~~~~~~

PAC methods
+++++++++++

If you don't want to use the :class:`tensorpac.Pac` class, you can also manually import the method of your choice
and use it on phase / amplitude to compute PAC.

.. currentmodule:: tensorpac.methods

.. autosummary::
   :toctree: generated/

   mvl
   kld
   hr
   ndpac
   ps
   gcpac

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


Miscellaneous
~~~~~~~~~~~~~

.. currentmodule:: tensorpac

.. autosummary::
   :toctree: generated/

   pac_vec
   pac_trivec
