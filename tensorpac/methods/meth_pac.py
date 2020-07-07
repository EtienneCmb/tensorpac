"""Individual methods for assessing PAC."""
import numpy as np
from scipy.special import erfinv

from functools import partial

from tensorpac.gcmi import nd_mi_gg


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
        method = 'Phase-Locking Value (PLV, Lachaux et al. 1999)'
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


###############################################################################
###############################################################################
#                                   PAC
###############################################################################
###############################################################################


def get_pac_fcn(idp, n_bins, p):
    """Get the function for computing Phase-Amplitude coupling."""
    if idp == 1:    # MVL
        return partial(mean_vector_length)
    elif idp == 2:  # KLD
        return partial(modulation_index, n_bins=n_bins)
    elif idp == 3:  # HR
        return partial(heights_ratio, n_bins=n_bins)
    elif idp == 4:  # ndPAC
        return partial(norm_direct_pac, p=p)
    elif idp == 5:  # PLV
        return partial(phase_locking_value)
    elif idp == 6:  # GC
        return partial(gauss_cop_pac)
    else:
        raise ValueError(str(idp) + " is not recognized as a valid pac "
                         "method.")


def mean_vector_length(pha, amp):
    """Compute PAC using the Mean Vector Length (MVL).

    Parameters
    ----------
    pha, amp : array_like
        Respectively the arrays of phases of shape (n_pha, ..., n_times) and
        the array of amplitudes of shape (n_amp, ..., n_times).

    Returns
    -------
    pac : array_like
        Array of phase amplitude coupling of shape (n_amp, n_pha, ...)

    References
    ----------
    Canolty et al. 2006 :cite:`canolty2006high`
    """
    return np.abs(np.einsum('i...j, k...j->ik...', amp,
                            np.exp(1j * pha))) / pha.shape[-1]


def modulation_index(pha, amp, n_bins=18):
    """Compute PAC using the Modulation index (MI).

    The modulation index is obtained using the Kullback Leibler Distance which
    measures how much the distribution of binned amplitude differs from a
    uniform distribution.

    Parameters
    ----------
    pha, amp : array_like
        Respectively the arrays of phases of shape (n_pha, ..., n_times) and
        the array of amplitudes of shape (n_amp, ..., n_times).
    n_bins : int | 18
        Number of bins to binarize the amplitude according to phase intervals

    Returns
    -------
    pac : array_like
        Array of phase amplitude coupling of shape (n_amp, n_pha, ...)

    References
    ----------
    Tort et al. 2010 :cite:`tort2010measuring`
    """
    # Get the phase locked binarized amplitude :
    p_j = _kl_hr(pha, amp, n_bins)
    # Divide the binned amplitude by the mean over the bins :
    p_j /= p_j.sum(axis=0, keepdims=True)
    # Take the log of non-zero values :
    p_j = p_j * np.ma.log(p_j).filled(-np.inf)
    # Compute the PAC :
    pac = 1 + p_j.sum(axis=0) / np.log(n_bins)
    # Set distribution distances that are really closed to zero :
    pac[np.isinf(pac)] = 0.
    return pac


def heights_ratio(pha, amp, n_bins=18):
    """Compute PAC using the Heights ratio (HR).

    Parameters
    ----------
    pha, amp : array_like
        Respectively the arrays of phases of shape (n_pha, ..., n_times) and
        the array of amplitudes of shape (n_amp, ..., n_times).
    n_bins : int | 18
        Number of bins to binarize the amplitude according to phase intervals

    Returns
    -------
    pac : array_like
        Array of phase amplitude coupling of shape (n_amp, n_pha, ...)

    References
    ----------
    Lakatos et al. 2005 :cite:`lakatos2005oscillatory`
    """
    # Get the phase locked binarized amplitude :
    p_j = _kl_hr(pha, amp, n_bins)
    # Divide the binned amplitude by the mean over the bins :
    p_j /= p_j.sum(axis=0, keepdims=True)
    # Find (maxximum, minimum) of the binned distribution :
    h_max, h_min = p_j.max(axis=0), p_j.min(axis=0)
    # Compute pac :
    pac = (h_max - h_min) / h_max
    return pac


def _kl_hr(pha, amp, n_bins, mean_bins=True):
    """Binarize the amplitude according to phase values.

    This function is shared by the Kullback-Leibler Distance and the
    Height Ratio.
    """
    vecbin = np.linspace(-np.pi, np.pi, n_bins + 1)
    phad = np.digitize(pha, vecbin) - 1

    abin = []
    for i in np.unique(phad):
        # Find where phase take vecbin values :
        idx = phad == i
        m = idx.sum() if mean_bins else 1.
        # Take the sum of amplitude inside the bin :
        abin_pha = np.einsum('i...j, k...j->ik...', amp, idx) / m
        abin.append(abin_pha)

    return np.array(abin)


def norm_direct_pac(pha, amp, p=.05):
    """Compute PAC using the Normalized direct Pac (ndPAC).

    Parameters
    ----------
    pha, amp : array_like
        Respectively the arrays of phases of shape (n_pha, ..., n_times) and
        the array of amplitudes of shape (n_amp, ..., n_times).
    p : float | .05
        P-value to use for thresholding. Sub-threshold PAC values
        will be set to 0. To disable this behavior (no masking), use ``p=1`` or
        ``p=None``.

    Returns
    -------
    pac : array_like
        Array of phase amplitude coupling of shape (n_amp, n_pha, ...)

    References
    ----------
    Ozkurt et al. :cite:`ozkurt2012statistically`
    """
    npts = amp.shape[-1]
    # Normalize amplitude :
    # Use the sample standard deviation, as in original Matlab code from author
    amp = np.subtract(amp, np.mean(amp, axis=-1, keepdims=True))
    amp = np.divide(amp, np.std(amp, ddof=1, axis=-1, keepdims=True))
    # Compute pac :
    pac = np.abs(np.einsum('i...j, k...j->ik...', amp, np.exp(1j * pha)))

    if p == 1. or p is None:
        # No thresholding
        return pac / npts

    s = pac**2
    pac /= npts
    # Set to zero non-significant values:
    xlim = npts * erfinv(1 - p)**2
    pac[s <= 2 * xlim] = 0.
    return pac


def phase_locking_value(pha, pha_amp):
    """Compute PAC using the Phase Locking-Value (PLV).

    In order to measure the phase locking value, the phase of the amplitude of
    the higher-frequency signal must be provided, and not the amplitude as in
    most other PAC functions.

    Parameters
    ----------
    pha, pha_amp : array_like
        Respectively the arrays of phases of shape (n_pha, ..., n_times) for
        the lower frequency and the array of phase of the amplitude signal of
        shape (n_pha_amp, ..., n_times) for the higher frequency.

    Returns
    -------
    pac : array_like
        Array of phase amplitude coupling of shape (n_pha_amp, n_pha, ...)

    References
    ----------
    Lachaux et al. 1999, :cite:`lachaux1999measuring`,
    Penny et al. 2008 :cite:`penny2008testing`
    """
    pac = np.einsum('i...j, k...j->ik...', np.exp(-1j * pha_amp),
                    np.exp(1j * pha))
    return np.abs(pac) / pha.shape[-1]


def gauss_cop_pac(pha, amp):
    """Compute PAC using the Gaussian Copula PAC (gcPac).

    This function assumes that phases and amplitudes have already been
    prepared i.e. phases should be represented in a unit circle
    (np.c_[np.sin(pha), np.cos(pha)]) and both inputs should also have been
    copnormed.

    Parameters
    ----------
    pha, amp : array_like
        Respectively the arrays of phases of shape (n_pha, ..., n_times) and
        the array of amplitudes of shape (n_amp, ..., n_times).

    Returns
    -------
    pac : array_like
        Array of phase amplitude coupling of shape (n_amp, n_pha, ...)

    References
    ----------
    Ince et al 2017. :cite:`ince2017statistical`
    """
    # prepare the shape of gcpac
    n_pha, n_amp = pha.shape[0], amp.shape[0]
    pha_sh = list(pha.shape[:-2])
    gc = np.zeros([n_amp] + pha_sh, dtype=float)
    # compute mutual information
    for p in range(n_pha):
        for a in range(n_amp):
            gc[a, p, ...] = nd_mi_gg(pha[p, ...], amp[a, ...])
    return gc
