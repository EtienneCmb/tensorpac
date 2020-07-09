"""Switch and utility functions for PAC methods."""
import numpy as np
from functools import partial


def get_pac_fcn(idp, n_bins, p, implementation="tensor", full=False):
    """Get the function for computing Phase-Amplitude coupling.

    This function also allow to switch between Tensor / Numba implementations
    of most of the functions.
    """
    n_bins, p = np.int64(n_bins), np.float64(p)
    assert implementation in ['tensor', 'numba']
    if implementation is 'tensor':
        from tensorpac.methods.meth_pac import (
            mean_vector_length, modulation_index, heights_ratio,
            norm_direct_pac, phase_locking_value, gauss_cop_pac)
        METH = {
            1: partial(mean_vector_length),
            2: partial(modulation_index, n_bins=n_bins),
            3: partial(heights_ratio, n_bins=n_bins),
            4: partial(norm_direct_pac, p=p),
            5: partial(phase_locking_value),
            6: partial(gauss_cop_pac)}
    elif implementation is 'numba':
        from tensorpac.methods.meth_pac_nb import (
            mean_vector_length_nb, modulation_index_nb, heights_ratio_nb,
            norm_direct_pac_nb, phase_locking_value_nb)
        # Gaussian-copula PAC can't be compiled with Numba. Hence, we force
        # using the Tensor implementation
        from tensorpac.methods.meth_pac import gauss_cop_pac
        METH = {
            1: partial(mean_vector_length_nb),
            2: partial(modulation_index_nb, n_bins=n_bins),
            3: partial(heights_ratio_nb, n_bins=n_bins),
            4: partial(norm_direct_pac_nb, p=p),
            5: partial(phase_locking_value_nb),
            6: partial(gauss_cop_pac)}

    if full:
        return METH
    else:
        return METH[idp]


def pacstr(idpac):
    """Return correspond methods string."""
    # Pac methods :
    if idpac[0] == 1:
        method = 'Mean Vector Length (MVL, Canolty et al. 2006)'
    elif idpac[0] == 2:
        method = 'Modulation Index (MI, Tort et al. 2010)'
    elif idpac[0] == 3:
        method = 'Heights ratio (HR, Lakatos et al. 2005)'
    elif idpac[0] == 4:
        method = 'Normalized Direct Pac (ndPac, Ozkurt et al. 2012)'
    elif idpac[0] == 5:
        method = 'Phase-Locking Value (PLV, Penny et al. 2008)'
    elif idpac[0] == 6:
        method = 'Gaussian Copula PAC (gcPac)'
    else:
        raise ValueError("No corresponding pac method.")

    # Surrogate method :
    if idpac[1] == 0:
        suro = 'No surrogates'
    elif idpac[1] == 1:
        suro = 'Permute phase across trials (Tort et al. 2010)'
    elif idpac[1] == 2:
        suro = 'Swap amplitude time blocks (Bahramisharif et al. 2013)'
    elif idpac[1] == 3:
        suro = 'Time lag (Canolty et al. 2006)'
    else:
        raise ValueError("No corresponding surrogate method.")

    # Normalization methods :
    if idpac[2] == 0:
        norm = 'No normalization'
    elif idpac[2] == 1:
        norm = 'Substract the mean of surrogates'
    elif idpac[2] == 2:
        norm = 'Divide by the mean of surrogates'
    elif idpac[2] == 3:
        norm = 'Substract then divide by the mean of surrogates'
    elif idpac[2] == 4:
        norm = "Substract the mean and divide by the deviation of the " + \
               "surrogates"
    else:
        raise ValueError("No corresponding normalization method.")

    return method, suro, norm
