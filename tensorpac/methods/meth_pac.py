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
        method = 'Kullback-Leiber Distance (KLD, Tort et al. 2010)'
    elif idpac[0] == 3:
        method = 'Heights ratio (HR, Lakatos et al. 2005)'
    elif idpac[0] == 4:
        method = 'ndPac (Ozkurt et al. 2012)'
    elif idpac[0] == 5:
        method = 'Phase-Locking Value (Lachaux et al. 1999)'
    elif idpac[0] == 6:
        method = 'Gaussian Copula PAC (Ince et al. 2017)'
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
        return partial(mvl)
    elif idp == 2:  # KLD
        return partial(kld, n_bins=n_bins)
    elif idp == 3:  # HR
        return partial(hr, n_bins=n_bins)
    elif idp == 4:  # ndPAC
        return partial(ndpac, p=p)
    elif idp == 5:  # PLV
        return partial(plv)
    elif idp == 6:  # GC
        return partial(gcpac)
    else:
        raise ValueError(str(idp) + " is not recognized as a valid pac "
                         "method.")


def mvl(pha, amp):
    """Mean Vector Length.

    Adapted from :cite:`canolty2006high`

    Parameters
    ----------
    pha, amp : array_like
        Respectively the arrays of phases of shape (n_pha, ..., n_times) and
        the array of amplitudes of shape (n_amp, ..., n_times).

    Returns
    -------
    pac : array_like
        Array of phase amplitude coupling of shape (n_amp, n_pha, ...)
    """
    return np.abs(np.einsum('i...j, k...j->ik...', amp,
                            np.exp(1j * pha))) / pha.shape[-1]


def kld(pha, amp, n_bins=18):
    """Kullback Leibler Distance.

    Adapted from :cite:`tort2010measuring`

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


def hr(pha, amp, n_bins=18):
    """Heights ratio.

    Adapted from :cite:`lakatos2005oscillatory`

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


def ndpac(pha, amp, p=.05):
    """Normalized direct Pac.

    Adapted from :cite:`ozkurt2012statistically`

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


def plv(pha, pha_amp):
    """Phase Locking-Value.

    In order to measure the phase locking value, the phase of the amplitude of
    the higher-frequency signal must be provided, and not the amplitude as in
    most other PAC functions. Adapted from
    :cite:`lachaux1999measuring,penny2008testing`

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
    """
    pac = np.einsum('i...j, k...j->ik...', np.exp(-1j * pha_amp),
                    np.exp(1j * pha))
    return np.abs(pac) / pha.shape[-1]


def gcpac(pha, amp):
    """Gaussian Copula Phase-amplitude coupling.

    This function assumes that phases and amplitudes have already been
    prepared i.e. phases should be represented in a unit circle
    (np.c_[np.sin(pha), np.cos(pha)]) and both inputs should also have been
    copnormed. Adapted from :cite:`ince2017statistical`

    Parameters
    ----------
    pha, amp : array_like
        Respectively the arrays of phases of shape (n_pha, ..., n_times) and
        the array of amplitudes of shape (n_amp, ..., n_times).

    Returns
    -------
    pac : array_like
        Array of phase amplitude coupling of shape (n_amp, n_pha, ...)
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
