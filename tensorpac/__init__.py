"""
PHASE-AMPLITUDE TOOLBOX
=======================

Tensorpac is an open-source Python toolbox designed for computing
Phase-Amplitude Coupling. 
"""
from .pac import Pac
from .utils import PacSignals, PacVec, PacTriVec

__all__ = ['Pac', 'PacSignals', 'PacVec', 'PacTriVec']
