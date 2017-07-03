"""
PHASE-AMPLITUDE TOOLBOX
=======================

Tensorpac is an open-source Python toolbox designed for computing
Phase-Amplitude Coupling.
"""
from .pac import Pac
from .utils import pac_signals, pac_vec, pac_trivec

__all__ = ('Pac', 'pac_signals', 'pac_vec', 'pac_trivec')
