"""Main PAC methods.

This file include the following methods :
- Mean Vector Length (Canolty, 2006)
- Kullback Leibler Distance (Tort, 2010)
- Heights Ratio (Lakatos, 2005)
- Normalized direct Pac (Ozkurt, 2012)
- Phase Synchrony (Penny, 2008; Cohen, 2008)
"""

import numpy as np
from scipy.special import erfinv

__all__ = ('compute_pac')


def compute_pac(pha, amp, idp, nbins, p, optimize):
    """Copute real Phase-Amplitude coupling.

    Each method take at least a pha and amp array with the respective
    dimensions:
    pha.shape = (npha, ..., npts)
    amp.shape = (namp, ..., npts)
    And each method should return a (namp, npha, ...)
    """
    if idp == 1:  # Mean Vector Length (Canolty, 2006)
        return mvl(pha, amp, optimize)

    elif idp == 2:  # Kullback-Leiber distance (Tort, 2010)
        return kld(pha, amp, nbins, optimize)

    elif idp == 3:  # Heights ratio (Lakatos, 2005)
        return hr(pha, amp, nbins, optimize)

    elif idp == 4:  # ndPac (Ozkurt, 2012)
        return ndpac(pha, amp, p, optimize)

    elif idp == 5:  # Phase-Synchrony (Penny, 2008; Cohen, 2008)
        return ps(pha, amp, optimize)

    else:
        raise ValueError(str(idp) + " is not recognized as a valid pac "
                         "method.")


def mvl(pha, amp, optimize):
    """Mean Vector Length (Canolty, 2006).

    Parameters
    ----------
    pha : array_like
        Array of phases of shapes (npha, ..., npts)

    amp : array_like
        Array of amplitudes of shapes (namp, ..., npts)

    Returns
    -------
    pac : array_like
        PAC of shape (npha, namp, ...)
    """
    # Number of time points :
    npts = pha.shape[-1]
    return np.abs(np.einsum('i...j, k...j->ik...', amp, np.exp(1j * pha),
                            optimize=optimize)) / npts


def kld(pha, amp, nbins, optimize):
    """Kullback Leibler Distance (Tort, 2010).

    Parameters
    ----------
    pha : array_like
        Array of phases of shapes (npha, ..., npts)

    amp : array_like
        Array of amplitudes of shapes (namp, ..., npts)

    nbins : int
        Number of bins in which the phase in cut in bins.

    Returns
    -------
    pac : array_like
        PAC of shape (npha, namp, ...)
    """
    # Get the phase locked binarized amplitude :
    p_j = _kl_hr(pha, amp, nbins, optimize)
    # Divide the binned amplitude by the mean over the bins :
    p_j /= p_j.sum(axis=0, keepdims=True)
    # Take the log of non-zero values :
    p_j = p_j * np.ma.log(p_j).filled(-np.inf)
    # Compute the PAC :
    pac = 1 + p_j.sum(axis=0) / np.log(nbins)
    # Set distribution distances that are really closed to zero :
    pac[np.isinf(pac)] = 0.
    return pac


def hr(pha, amp, nbins, optimize):
    """Pac heights ratio (Lakatos, 2005).

    Parameters
    ----------
    pha : array_like
        Array of phases of shapes (npha, ..., npts)

    amp : array_like
        Array of amplitudes of shapes (namp, ..., npts)

    nbins : int
        Number of bins in which the phase in cut in bins.

    Returns
    -------
    pac : array_like
        PAC of shape (npha, namp, ...)
    """
    # Get the phase locked binarized amplitude :
    p_j = _kl_hr(pha, amp, nbins, optimize)
    # Divide the binned amplitude by the mean over the bins :
    p_j /= p_j.sum(axis=0, keepdims=True)
    # Find (maxximum, minimum) of the binned distribution :
    h_max, h_min = p_j.max(axis=0), p_j.min(axis=0)
    # Compute pac :
    pac = (h_max - h_min) / h_max
    return pac


def _kl_hr(pha, amp, nbins, optimize):
    """Binarize the amplitude according to phase values.

    This function is shared by the Kullback-Leibler Distance and the
    Height Ratio.
    """
    vecbin = np.linspace(-np.pi, np.pi, nbins + 1)
    phad = np.digitize(pha, vecbin) - 1

    abin = []
    for i in np.unique(phad):
        # Find where phase take vecbin values :
        idx = phad == i
        # Take the sum of amplitude inside the bin :
        abin_pha = np.einsum('i...j, k...j->ik...', amp, idx,
                             optimize=optimize)
        abin.append(abin_pha)

    return np.array(abin)


def ndpac(pha, amp, p, optimize):
    """Normalized direct Pac (Ozkurt, 2012).

    Parameters
    ----------
    pha : array_like
        Array of phases of shapes (npha, ..., npts)

    amp : array_like
        Array of amplitudes of shapes (namp, ..., npts)

    p : float
        The p-value to use.

    Returns
    -------
    pac : array_like
        PAC of shape (npha, namp, ...)
    """
    npts = amp.shape[-1]
    # Normalize amplitude :
    np.subtract(amp, np.mean(amp, axis=-1, keepdims=True), out=amp)
    np.divide(amp, np.std(amp, axis=-1, keepdims=True), out=amp)
    # Compute pac :
    pac = np.abs(np.einsum('i...j, k...j->ik...', amp, np.exp(1j * pha),
                           optimize=optimize))
    pac *= pac / npts
    # Set to zero non-significant values:
    xlim = erfinv(1 - p)**2
    pac[pac <= 2 * xlim] = 0.
    return pac


def ps(pha, amp, optimize):
    """Phase Synchrony (Penny, 2008; Cohen, 2008).

    Parameters
    ----------
    pha : array_like
        Array of phases of shapes (npha, ..., npts)

    amp : array_like
        Array of amplitudes of shapes (namp, ..., npts)

    Returns
    -------
    pac : array_like
        PAC of shape (npha, namp, ...)
    """
    # Number of time points :
    npts = pha.shape[-1]
    pac = np.einsum('i...j, k...j->ik...', np.exp(-1j * amp), np.exp(1j * pha),
                    optimize=optimize)
    return np.abs(pac) / npts
