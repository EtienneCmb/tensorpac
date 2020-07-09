"""Numba implementation of some PAC functions."""
import numpy as np
from scipy.special import erfinv


# if Numba not installed, this section should return a Numba-free jit wrapper
try:
    import numba
    def jit(signature=None, nopython=True, nogil=True, fastmath=True,  # noqa
            cache=True, **kwargs):
        return numba.jit(signature_or_function=signature, cache=cache,
                         nogil=nogil, fastmath=fastmath, nopython=nopython,
                         **kwargs)
except:
    def jit(*args, **kwargs):  # noqa
        def _jit(func):
            return func
        return _jit


@jit("f8[:,:,:](f8[:,:,:], f8[:,:,:])")
def mean_vector_length_nb(pha, amp):
    """Numba-based Mean Vector Length (MVL).

    Parameters
    ----------
    pha, amp : array_like
        Respectively the arrays of phases of shape (n_pha, n_epochs, n_times)
        and the array of amplitudes of shape (n_amp, n_epochs, n_times). Both
        arrays should be of type float64 (np.float64)

    Returns
    -------
    pac : array_like
        Array of phase amplitude coupling of shape (n_amp, n_pha, n_epochs)

    References
    ----------
    Canolty et al. 2006 :cite:`canolty2006high`
    """
    n_pha, n_epochs, n_times = pha.shape
    n_amp, _, _ = amp.shape
    pac = np.zeros((n_amp, n_pha, n_epochs), dtype=np.float64)
    # single conversion
    exp_pha = np.exp(1j * pha)
    amp_comp = amp.astype(np.complex128)
    for a in range(n_amp):
        for p in range(n_pha):
            for tr in range(n_epochs):
                _pha = np.ascontiguousarray(exp_pha[p, tr, :])
                _amp = np.ascontiguousarray(amp_comp[a, tr, :])
                pac[a, p, tr] = abs(np.dot(_amp, _pha))
    pac /= n_times

    return pac


@jit("f8[:](f8[:], f8[:], u8, b1)")
def _kl_hr_nb(pha, amp, n_bins=18, mean_bins=True):
    """Binarize the amplitude according to phase values.

    This function is shared by the Kullback-Leibler Distance and the
    Height Ratio.
    """
    vecbin = np.linspace(-np.pi, np.pi, n_bins + 1)
    phad = np.digitize(pha, vecbin) - 1
    u_phad = np.unique(phad)

    abin = np.zeros((len(u_phad)), dtype=np.float64)
    for n_i, i in enumerate(u_phad):
        # find where phase take vecbin values
        idx = np.ascontiguousarray((phad == i).astype(np.float64))
        m = idx.sum() if mean_bins else 1.
        # take the sum of amplitude inside the bin
        abin[n_i] = np.dot(np.ascontiguousarray(amp), idx) / m

    return abin


@jit("f8[:,:,:](f8[:,:,:], f8[:,:,:], u8)")
def modulation_index_nb(pha, amp, n_bins=18):
    """Numba-based Modulation index (MI).

    The modulation index is obtained using the Kullback Leibler Distance which
    measures how much the distribution of binned amplitude differs from a
    uniform distribution.

    Parameters
    ----------
    pha, amp : array_like
        Respectively the arrays of phases of shape (n_pha, n_epochs, n_times)
        and the array of amplitudes of shape (n_amp, n_epochs, n_times). Both
        arrays should be of type float64 (np.float64)
    n_bins : int | 18
        Number of bins to binarize the amplitude according to phase intervals
        (should be np.int64)

    Returns
    -------
    pac : array_like
        Array of phase amplitude coupling of shape (n_amp, n_pha, ...)

    References
    ----------
    Tort et al. 2010 :cite:`tort2010measuring`
    """
    n_pha, n_epochs, n_times = pha.shape
    n_amp, _, _ = amp.shape
    pac = np.zeros((n_amp, n_pha, n_epochs), dtype=np.float64)
    bin_log = np.log(n_bins)

    for a in range(n_amp):
        for p in range(n_pha):
            for tr in range(n_epochs):
                # select phase and amplitude
                _pha = np.ascontiguousarray(pha[p, tr, :])
                _amp = np.ascontiguousarray(amp[a, tr, :])
                # get the probability of each amp bin
                p_j = _kl_hr_nb(_pha, _amp, n_bins=n_bins, mean_bins=True)
                p_j /= p_j.sum()
                # log it (only if strictly positive)
                if np.all(p_j > 0.):
                    p_j *= np.log(p_j)
                    # compute the PAC
                    pac[a, p, tr] = 1. + p_j.sum() / bin_log
                else:
                    pac[a, p, tr] = 0.

    return pac


@jit("f8[:,:,:](f8[:,:,:], f8[:,:,:], u8)")
def heights_ratio_nb(pha, amp, n_bins=18):
    """Numba-based Heights ratio (HR).

    Parameters
    ----------
    pha, amp : array_like
        Respectively the arrays of phases of shape (n_pha, n_epochs, n_times)
        and the array of amplitudes of shape (n_amp, n_epochs, n_times). Both
        arrays should be of type float64 (np.float64)
    n_bins : int | 18
        Number of bins to binarize the amplitude according to phase intervals
        (should be np.int64)

    Returns
    -------
    pac : array_like
        Array of phase amplitude coupling of shape (n_amp, n_pha, ...)

    References
    ----------
    Lakatos et al. 2005 :cite:`lakatos2005oscillatory`
    """
    n_pha, n_epochs, n_times = pha.shape
    n_amp, _, _ = amp.shape
    pac = np.zeros((n_amp, n_pha, n_epochs), dtype=np.float64)

    for a in range(n_amp):
        for p in range(n_pha):
            for tr in range(n_epochs):
                # select phase and amplitude
                _pha = np.ascontiguousarray(pha[p, tr, :])
                _amp = np.ascontiguousarray(amp[a, tr, :])
                # get the probability of each amp bin
                p_j = _kl_hr_nb(_pha, _amp, n_bins=n_bins, mean_bins=True)
                p_j /= p_j.sum()
                # find (maximum, minimum) of the binned distribution
                h_max, h_min = np.max(p_j), np.min(p_j)
                # compute the PAC
                pac[a, p, tr] = (h_max - h_min) / h_max

    return pac


def phase_locking_value_nb(pha, pha_amp):
    """Numba-based Phase Locking-Value (PLV).

    In order to measure the phase locking value, the phase of the amplitude of
    the higher-frequency signal must be provided, and not the amplitude as in
    most other PAC functions.

    Parameters
    ----------
    pha, pha_amp : array_like
        Respectively the arrays of phases of shape (n_pha, n_epochs, n_times)
        for the lower frequency and the array of phase of the amplitude signal
        of shape (n_pha_amp, n_epochs, n_times) for the higher frequency. Both
        arrays should be of type float64 (np.float64)

    Returns
    -------
    pac : array_like
        Array of phase amplitude coupling of shape (n_pha_amp, n_pha, ...)

    References
    ----------
    Penny et al. 2008 :cite:`penny2008testing`, Lachaux et al. 1999
    :cite:`lachaux1999measuring`
    """
    n_pha, n_epochs, n_times = pha.shape
    n_amp, _, _ = pha_amp.shape
    pac = np.zeros((n_amp, n_pha, n_epochs), dtype=np.float64)
    # single conversion
    exp_pha = np.exp(1j * pha)
    exp_pha_amp = np.exp(-1j * pha_amp)
    for a in range(n_amp):
        for p in range(n_pha):
            for tr in range(n_epochs):
                _pha = exp_pha[p, tr, :]
                _pha_amp = exp_pha_amp[a, tr, :]
                pac[a, p, tr] = abs(np.dot(_pha, _pha_amp))
    pac /= n_times

    return pac

"""
I don't think this function can be entirely compiled with Numba because of two
issues :

    * Numba supports the mean / std but not across a specific axis
    * erfinv is a special function of scipy that don't seems to be supported
      for the moment

Therefore, the beginning and the end of the function are tensor-based while the
core function that computes PAC is the Numba compliant MVL.
"""
def norm_direct_pac_nb(pha, amp, p=.05):
    """Numba-based Normalized direct Pac (ndPAC).

    Parameters
    ----------
    pha, amp : array_like
        Respectively the arrays of phases of shape (n_pha, n_epochs, n_times)
        and the array of amplitudes of shape (n_amp, n_epochs, n_times). Both
        arrays should be of type float64 (np.float64)
    p : float | .05
        P-value to use for thresholding. Sub-threshold PAC values
        will be set to 0. To disable this behavior (no masking), use ``p=1`` or
        ``p=None``. Should be a np.float64

    Returns
    -------
    pac : array_like
        Array of phase amplitude coupling of shape (n_amp, n_pha, ...)

    References
    ----------
    Ozkurt et al. :cite:`ozkurt2012statistically`
    """
    n_times = pha.shape[-1]
    # z-score normalization to approximate assumptions
    amp = np.subtract(amp, np.mean(amp, axis=-1, keepdims=True))
    amp = np.divide(amp, np.std(amp, ddof=1, axis=-1, keepdims=True))
    # compute pac using MVL (need to remultiply by n_times)
    pac = mean_vector_length_nb(pha, amp) * n_times

    # no thresholding
    if p == 1. or p is None:
        return pac / n_times

    s = pac ** 2
    pac /= n_times
    # set to zero non-significant values
    xlim = n_times * erfinv(1 - p) ** 2
    pac[s <= 2 * xlim] = 0.
    return pac
