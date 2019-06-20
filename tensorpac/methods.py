"""Main PAC methods."""
import numpy as np
from scipy.special import erfinv

from functools import partial
from joblib import Parallel, delayed

from tensorpac.gcmi import nd_mi_gg  # copnorm
from tensorpac.config import JOBLIB_CFG


def pacstr(idpac):
    """Return correspond methods string."""
    # Pac methods :
    if idpac[0] == 1:
        method = 'Mean Vector Length (MVL, Canolty, 2006)'
    elif idpac[0] == 2:
        method = 'Kullback-Leiber Distance (KLD, Tort, 2010)'
    elif idpac[0] == 3:
        method = 'Heights ratio (HR, Lakatos, 2005)'
    elif idpac[0] == 4:
        method = 'ndPac (Ozkurt, 2012)'
    elif idpac[0] == 5:
        method = 'Phase-Synchrony (Cohen, 2008; Penny, 2008)'
    elif idpac[0] == 6:
        method = 'Gaussian Copula PAC'
    else:
        raise ValueError("No corresponding pac method.")

    # Surrogate method :
    if idpac[1] == 0:
        suro = 'No surrogates'
    elif idpac[1] == 1:
        suro = 'Permute phase across trials'
    elif idpac[1] == 2:
        suro = 'Swap amplitude time blocks'
    elif idpac[1] == 3:
        suro = 'Time lag'
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
    """Mean Vector Length (Canolty, 2006)."""
    return np.abs(np.einsum('i...j, k...j->ik...', amp,
                            np.exp(1j * pha))) / pha.shape[-1]


def kld(pha, amp, n_bins=18):
    """Kullback Leibler Distance (Tort, 2010)."""
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
    """Pac heights ratio (Lakatos, 2005)."""
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
    """Normalized direct Pac (Ozkurt, 2012)."""
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
    """Phase Synchrony (Penny, 2008; Cohen, 2008)."""
    pac = np.einsum('i...j, k...j->ik...', np.exp(-1j * amp), np.exp(1j * pha))
    return np.abs(pac) / pha.shape[-1]


def gcpac(pha, amp):
    """Gaussian Copula."""
    # prepare the shape of gcpac
    n_pha, n_amp = pha.shape[0], amp.shape[0]
    pha_sh = list(pha.shape[:-1])
    gc = np.zeros([n_amp] + pha_sh, dtype=float)
    # concatenate sine and cosine
    sco = np.stack([np.sin(pha), np.cos(pha)], axis=-2)
    amp = amp[..., np.newaxis, :]
    # copnorm the data
    # amp = np.apply_along_axis(copnorm, -1, amp)
    # sco = np.apply_along_axis(copnorm, -1, sco)
    # compute mutual information
    for p in range(n_pha):
        for a in range(n_amp):
            gc[a, p, ...] = nd_mi_gg(sco[p, ...], amp[a, ...], mvaxis=-2,
                                     traxis=-1, biascorrect=False)
    return gc


###############################################################################
###############################################################################
#                                 SURROGATES
###############################################################################
###############################################################################

def compute_surrogates(pha, amp, ids, fcn, n_perm, n_jobs):
    """Compute surrogates using tensors and parallel computing."""
    if ids == 0:
        return None
    else:
        fcn_p = {1: swap_pha_amp, 2: swap_blocks, 3: time_lag}[ids]
    s = Parallel(n_jobs=n_jobs, **JOBLIB_CFG)(delayed(fcn)(
        *fcn_p(pha, amp)) for k in range(n_perm))
    return np.array(s)


def swap_pha_amp(pha, amp):
    """Swap phase / amplitude trials (Tort, 2010)."""
    tr_ = np.random.permutation(pha.shape[1])
    return pha[:, tr_, ...], amp


def swap_blocks(pha, amp):
    """Swap amplitudes time blocks (Bahramisharif, 2013)."""
    # random cutting point along time axis
    cut_at = np.random.randint(1, amp.shape[-1], (1,))
    # Split amplitude across time into two parts :
    ampl = np.array_split(amp, cut_at, axis=-1)
    # Revered elements :
    ampl.reverse()
    return pha, np.concatenate(ampl, axis=-1)


def time_lag(pha, amp):
    """Introduce a time lag on phase series (Canolty et al. 2006)."""
    shift = np.random.randint(pha.shape[-1])
    return np.roll(pha, shift, axis=-1), amp


###############################################################################
###############################################################################
#                                 NORMALIZATION
###############################################################################
###############################################################################


def normalize(pac, s_mean, s_std, idn):
    """PAC normalization."""
    if idn == 1:  # Substraction
        pac -= s_mean
    elif idn == 2:  # Divide
        pac /= s_mean
    elif idn == 3:  # Substract then divide
        pac -= s_mean
        pac /= s_mean
    elif idn == 4:  # Z-score
        pac -= s_mean
        pac /= s_std
