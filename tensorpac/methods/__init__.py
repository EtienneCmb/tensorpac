"""Tensorpac methods."""
from .meth_pac import (
    pacstr, get_pac_fcn, mean_vector_length, modulation_index, heights_ratio,
    norm_direct_pac, phase_locking_value, gauss_cop_pac)  # noqa
from .meth_erpac import (erpac, ergcpac, _ergcpac_perm)  # noqa
from .meth_pp import (preferred_phase)  # noqa
from .meth_surrogates import (compute_surrogates, swap_pha_amp, swap_blocks,  # noqa
                              time_lag, normalize)
