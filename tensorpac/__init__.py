"""
PHASE-AMPLITUDE TOOLBOX
=======================

Tensorpac is an open-source Python toolbox designed for computing
Phase-Amplitude Coupling.
"""
import logging

from tensorpac.pac import Pac  # noqa
from tensorpac.io import set_log_level
from tensorpac.utils import (pac_signals_wavelet, pac_signals_tort, pac_vec,  # noqa
                             pac_trivec)
# Set 'info' as the default logging level
logger = logging.getLogger('brainets')
set_log_level('info')

__version__ = "0.5.6"
