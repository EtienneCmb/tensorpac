"""
Tensorpac
=========

Tensorpac is an open-source Python toolbox designed for computing
Phase-Amplitude Coupling.
"""
import logging

from tensorpac import methods, signals, utils, stats  # noqa
from tensorpac.pac import (Pac, EventRelatedPac, PreferredPhase)  # noqa
from tensorpac.io import set_log_level
# Set 'info' as the default logging level
logger = logging.getLogger('brainets')
set_log_level('info')

__version__ = "0.6.5"
