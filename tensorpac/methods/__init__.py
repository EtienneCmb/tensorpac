"""Tensorpac methods."""
from .meth_pac import (pacstr, get_pac_fcn, mvl, kld, hr, ndpac, ps, gcpac)  # noqa
from .meth_erpac import (erpac, ergcpac, _ergcpac_perm)  # noqa
from .meth_pp import (preferred_phase)  # noqa
from .meth_surrogates import (compute_surrogates, swap_pha_amp, swap_blocks,  # noqa
                              time_lag, normalize)
