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
        method = 'Phase-Synchrony (Cohen et al. 2008; Penny et al. 2008)'
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
    if idp == 1:  # Mean Vector Length (Canolty, 2006)
        return partial(mvl)
    elif idp == 2:  # Kullback-Leiber distance (Tort, 2010)
        return partial(kld, n_bins=n_bins)
    elif idp == 3:  # Heights ratio (Lakatos, 2005)
        return partial(hr, n_bins=n_bins)
    elif idp == 4:  # ndPac (Ozkurt, 2012)
        return partial(ndpac, p=p)
    elif idp == 5:  # Phase-Synchrony (Penny, 2008; Cohen, 2008)
        return partial(ps)
    elif idp == 6:  # Gaussian-Copula
        return partial(gcpac)
    else:
        raise ValueError(str(idp) + " is not recognized as a valid pac "
                         "method.")


def mvl(pha, amp):
    """Mean Vector Length.

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
    Canolty RT (2006) High Gamma Power Is Phase-Locked to Theta. science
    1128115:313.
    """
    return np.abs(np.einsum('i...j, k...j->ik...', amp,
                            np.exp(1j * pha))) / pha.shape[-1]


def kld(pha, amp, n_bins=18):
    """Kullback Leibler Distance.

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
    Tort ABL, Komorowski R, Eichenbaum H, Kopell N (2010) Measuring
    Phase-Amplitude Coupling Between Neuronal Oscillations of Different
    Frequencies. Journal of Neurophysiology 104:1195–1210.
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
    Lakatos P (2005) An Oscillatory Hierarchy Controlling Neuronal
    Excitability and Stimulus Processing in the Auditory Cortex. Journal of
    Neurophysiology 94:1904–1911.
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


def _kl_hr(pha, amp, n_bins):
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
        # Take the sum of amplitude inside the bin :
        abin_pha = np.einsum('i...j, k...j->ik...', amp, idx)
        abin.append(abin_pha)

    return np.array(abin)


def ndpac(pha, amp, p=.05):
    """Normalized direct Pac.

    Parameters
    ----------
    pha, amp : array_like
        Respectively the arrays of phases of shape (n_pha, ..., n_times) and
        the array of amplitudes of shape (n_amp, ..., n_times).
    p : float | .05
        P-value to use for thresholding

    Returns
    -------
    pac : array_like
        Array of phase amplitude coupling of shape (n_amp, n_pha, ...)

    References
    ----------
    Ozkurt TE (2012) Statistically Reliable and Fast Direct Estimation of
    Phase-Amplitude Cross-Frequency Coupling. Biomedical Engineering, IEEE
    Transactions on 59:1943–1950.
    """
    npts = amp.shape[-1]
    # Normalize amplitude :
    np.subtract(amp, np.mean(amp, axis=-1, keepdims=True), out=amp)
    np.divide(amp, np.std(amp, axis=-1, keepdims=True), out=amp)
    # Compute pac :
    pac = np.abs(np.einsum('i...j, k...j->ik...', amp, np.exp(1j * pha)))
    pac *= pac / npts
    # Set to zero non-significant values:
    xlim = erfinv(1 - p)**2
    pac[pac <= 2 * xlim] = 0.
    return pac


def ps(pha, amp):
    """Phase Synchrony (Penny, 2008; Cohen, 2008).

    In order to measure the phase synchrony, the phase of the amplitude must be
    provided.

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
    Penny WD, Duzel E, Miller KJ, Ojemann JG (2008) Testing for nested
    oscillation. Journal of Neuroscience Methods 174:50–61.
    Cohen MX, Elger CE, Fell J (2008) Oscillatory activity and phase amplitude
    coupling in the human medial frontal cortex during decision making. Journal
    of cognitive neuroscience 21:390–402.
    """
    pac = np.einsum('i...j, k...j->ik...', np.exp(-1j * amp), np.exp(1j * pha))
    return np.abs(pac) / pha.shape[-1]


def gcpac(pha, amp):
    """Gaussian Copula Phase-amplitude coupling.

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
    Ince RAA, Giordano BL, Kayser C, Rousselet GA, Gross J, Schyns PG (2017) A
    statistical framework for neuroimaging data analysis based on mutual
    information estimated via a gaussian copula: Gaussian Copula Mutual
    Information. Human Brain Mapping 38:1541–1573.
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
