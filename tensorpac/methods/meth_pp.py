"""Individual methods for assessing Preferred Phase."""
import numpy as np

from .meth_pac import _kl_hr


def preferred_phase(pha, amp, n_bins=18):
    """Compute the preferred phase.

    Parameters
    ----------
    pha, amp : array_like
        Respectively the phase of slower oscillations of shape
        (n_pha, n_epochs, n_times) and the amplitude of faster
        oscillations of shape (n_pha, n_epochs, n_times).
    n_bins : int | 72
        Number of bins for bining the amplitude according to phase
        slices.

    Returns
    -------
    binned_amp : array_like
        The binned amplitude according to the phase of shape
        (n_bins, n_amp, n_pha, n_epochs).
    pp : array_like
        The prefered phase where the amplitude is maximum of shape
        (namp, npha, n_epochs).
    polarvec : array_like
        The phase vector for the polar plot of shape (n_bins,)
    """
    # Bin the amplitude according to the phase :
    binned_amp = _kl_hr(pha, amp, n_bins)
    binned_amp /= binned_amp.sum(axis=0, keepdims=True)
    # Find the index where the amplitude is maximum over the bins :
    idxmax = binned_amp.argmax(axis=0)
    # Find the preferred phase :
    binsize = (2 * np.pi) / float(n_bins)
    vecbin = np.arange(-np.pi, np.pi, binsize) + binsize / 2
    pp = vecbin[idxmax]
    # Build the phase vector (polar plot) :
    polarvec = np.linspace(-np.pi, np.pi, binned_amp.shape[0])
    return binned_amp, pp, polarvec
