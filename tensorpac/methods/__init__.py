"""Tensorpac methods."""
from .meth_pac import (
    mean_vector_length, modulation_index, heights_ratio, norm_direct_pac,
    phase_locking_value, gauss_cop_pac)  # noqa
from .meth_pac_nb import (
    mean_vector_length_nb, modulation_index_nb, heights_ratio_nb,
    norm_direct_pac_nb, phase_locking_value_nb)  # noqa
from .meth_erpac import (erpac, ergcpac, _ergcpac_perm)  # noqa
from .meth_pp import (preferred_phase)  # noqa
from .meth_surrogates import (compute_surrogates, swap_pha_amp, swap_blocks,  # noqa
                              time_lag, normalize)
from .meth_switch import (pacstr, get_pac_fcn)  # noqa
