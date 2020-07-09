"""Individual methods for assessing PAC."""
import numpy as np
from scipy.special import erfinv

from tensorpac.gcmi import nd_mi_gg


def mean_vector_length(pha, amp):
    """Tensor-based Mean Vector Length (MVL).

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
    """Tensor-based Modulation index (MI).

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
    """Tensor-based Heights ratio (HR).

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
    """Tensor-based Normalized direct Pac (ndPAC).

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
    n_times = amp.shape[-1]
    # normalize amplitude
    # use the sample standard deviation, as in original matlab code from author
    amp = np.subtract(amp, np.mean(amp, axis=-1, keepdims=True))
    amp = np.divide(amp, np.std(amp, ddof=1, axis=-1, keepdims=True))
    # compute pac
    pac = np.abs(np.einsum('i...j, k...j->ik...', amp, np.exp(1j * pha)))

    # no thresholding
    if p == 1. or p is None:
        return pac / n_times

    s = pac ** 2
    pac /= n_times
    # set to zero non-significant values
    xlim = n_times * erfinv(1 - p) ** 2
    pac[s <= 2 * xlim] = 0.
    return pac


def phase_locking_value(pha, pha_amp):
    """Tensor-based Phase Locking-Value (PLV).

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
    Penny et al. 2008 :cite:`penny2008testing`, Lachaux et al. 1999
    :cite:`lachaux1999measuring`
    """
    pac = np.einsum('i...j, k...j->ik...', np.exp(-1j * pha_amp),
                    np.exp(1j * pha))
    return np.abs(pac) / pha.shape[-1]


def gauss_cop_pac(pha, amp):
    """Tensor-based Gaussian Copula PAC (gcPac).

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
