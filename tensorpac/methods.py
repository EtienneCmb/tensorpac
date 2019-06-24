"""Main PAC methods."""
import numpy as np
from scipy.special import erfinv
from scipy.stats import chi2

from functools import partial
from joblib import Parallel, delayed

from tensorpac.gcmi import nd_mi_gg, copnorm
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
            gc[a, p, ...] = nd_mi_gg(pha[p, ...], amp[a, ...], mvaxis=-2,
                                     traxis=-1, biascorrect=False)
    return gc


###############################################################################
###############################################################################
#                  EVENT RELATED PHASE AMPLITUDE COUPLING
###############################################################################
###############################################################################


def pearson(x, y, st='i...j, k...j->ik...'):
    """Pearson correlation for multi-dimensional arrays.

    Parameters
    ----------
    x, y : array_like
        Compute pearson correlation between the multi-dimensional arrays
        x and y.
    st : string | 'i..j, k..j->ik...'
        The string to pass to the np.einsum function.

    Returns
    -------
    cov: array_like
        The pearson correlation array.
    """
    n = x.shape[-1]
    # Distribution center :
    mu_x = x.mean(-1, keepdims=True)
    mu_y = y.mean(-1, keepdims=True)
    # Distribution deviation :
    s_x = x.std(-1, ddof=n - 1, keepdims=True)
    s_y = y.std(-1, ddof=n - 1, keepdims=True)
    # Compute correlation coefficient :
    cov = np.einsum(st, x, y)
    mu_xy = np.einsum(st, mu_x, mu_y)
    cov -= n * mu_xy
    cov /= np.einsum(st, s_x, s_y)
    return cov


def erpac(pha, amp):
    """Correlation coefficient between a circular and a linear random variable.

    Adapted from the function circ_corrcc Circular Statistics Toolbox for
    Matlab By Philipp Berens, 2009.

    Parameters
    ----------
    pha, amp : array_like
        Respectively the arrays of phases of shape (n_pha, ..., n_epochs) and
        the array of amplitudes of shape (n_amp, ..., n_epochs).

    Returns
    -------
    rho : array_like
        Array of correlation coefficients of shape (n_amp, n_pha, ...)
    pval : array_like
        Array of p-values of shape (n_amp, n_pha, ...).

    References
    ----------
    Voytek B, D’Esposito M, Crone N, Knight RT (2013) A method for
    event-related phase/amplitude coupling. NeuroImage 64:416–424.
    """
    # Move the trial axis to the end :
    pha = np.moveaxis(pha, 1, -1)
    amp = np.moveaxis(amp, 1, -1)
    # Compute correlation coefficient for sin and cos independently
    n = pha.shape[-1]
    sa, ca = np.sin(pha), np.cos(pha)
    rxs = pearson(amp, sa)
    rxc = pearson(amp, ca)
    rcs = pearson(sa, ca, st='i...j, k...j->i...')
    rcs = rcs[np.newaxis, ...]

    # Compute angular-linear correlation (equ. 27.47)
    rho = np.sqrt((rxc**2 + rxs**2 - 2 * rxc * rxs * rcs) / (1 - rcs**2))

    # Compute pvalue :
    pval = 1. - chi2.cdf(n * rho**2, 2)

    return rho, pval


def ergcpac(pha, amp):
    """Event Related PAC computed using the Gaussian Copula Mutual Information.

    Parameters
    ----------
    pha, amp : array_like
        Respectively the arrays of phases of shape (n_pha, n_times, n_epochs)
        and the array of amplitudes of shape (n_amp, n_times, n_epochs).

    Returns
    -------
    rho : array_like
        Array of correlation coefficients of shape (n_amp, n_pha, n_times)

    References
    ----------
    Ince RAA, Giordano BL, Kayser C, Rousselet GA, Gross J, Schyns PG (2017) A
    statistical framework for neuroimaging data analysis based on mutual
    information estimated via a gaussian copula: Gaussian Copula Mutual
    Information. Human Brain Mapping 38:1541–1573.
    """
    # Move the trial axis to the end :
    pha = np.moveaxis(pha, 1, -1)
    amp = np.moveaxis(amp, 1, -1)
    # get shapes
    n_pha, n_times, n_epochs = pha.shape
    n_amp = amp.shape[0]
    # conversion for computing mi
    sco = copnorm(np.stack([np.sin(pha), np.cos(pha)], axis=-2))
    amp = copnorm(amp)[..., np.newaxis, :]
    # compute mutual information across trials
    ergcpac = np.zeros((n_amp, n_pha, n_times))
    for a in range(n_amp):
        for p in range(n_pha):
            ergcpac[a, p, ...] = nd_mi_gg(sco[p, ...], amp[a, ...], mvaxis=-2,
                                          traxis=-1, biascorrect=False)
    return ergcpac


###############################################################################
###############################################################################
#                              PREFERRED PHASE
###############################################################################
###############################################################################


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
    """Swap phase / amplitude trials.

    Parameters
    ----------
    pha, amp : array_like
        Respectively the arrays of phases of shape (n_pha, ..., n_times) and
        the array of amplitudes of shape (n_amp, ..., n_times).

    Returns
    -------
    pha, amp : array_like
        The phase and amplitude to use to compute the distribution of
        permutations

    References
    ----------
    Tort ABL, Komorowski R, Eichenbaum H, Kopell N (2010) Measuring
    Phase-Amplitude Coupling Between Neuronal Oscillations of Different
    Frequencies. Journal of Neurophysiology 104:1195–1210.
    """
    tr_ = np.random.permutation(pha.shape[1])
    return pha[:, tr_, ...], amp


def swap_blocks(pha, amp):
    """Swap amplitudes time blocks (Bahramisharif, 2013).

    Parameters
    ----------
    pha, amp : array_like
        Respectively the arrays of phases of shape (n_pha, ..., n_times) and
        the array of amplitudes of shape (n_amp, ..., n_times).

    Returns
    -------
    pha, amp : array_like
        The phase and amplitude to use to compute the distribution of
        permutations

    References
    ----------
    Bahramisharif A, van Gerven MAJ, Aarnoutse EJ, Mercier MR, Schwartz TH,
    Foxe JJ, Ramsey NF, Jensen O (2013) Propagating Neocortical Gamma Bursts
    Are Coordinated by Traveling Alpha Waves. Journal of Neuroscience
    33:18849–18854.
    """
    # random cutting point along time axis
    cut_at = np.random.randint(1, amp.shape[-1], (1,))
    # Split amplitude across time into two parts :
    ampl = np.array_split(amp, cut_at, axis=-1)
    # Revered elements :
    ampl.reverse()
    return pha, np.concatenate(ampl, axis=-1)


def time_lag(pha, amp):
    """Introduce a time lag on phase series (Canolty et al. 2006).

    Parameters
    ----------
    pha, amp : array_like
        Respectively the arrays of phases of shape (n_pha, ..., n_times) and
        the array of amplitudes of shape (n_amp, ..., n_times).

    Returns
    -------
    pha, amp : array_like
        The phase and amplitude to use to compute the distribution of
        permutations

    References
    ----------
    Canolty RT (2006) High Gamma Power Is Phase-Locked to Theta. science
    1128115:313.
    """
    shift = np.random.randint(pha.shape[-1])
    return np.roll(pha, shift, axis=-1), amp


###############################################################################
###############################################################################
#                                 NORMALIZATION
###############################################################################
###############################################################################


def normalize(idn, pac, surro):
    """Normalize the phase amplitude coupling.

    This function performs inplace normalization (i.e. without copy of array)

    Parameters
    ----------
    idn : int
        Normalization method to use :

            * 1 : substract the mean of surrogates
            * 2 : divide by the mean of surrogates
            * 3 : substract then divide by the mean of surrogates
            * 4 : substract the mean then divide by the deviation of surrogates
    pac : array_like
        Array of phase amplitude coupling of shape (n_amp, n_pha, ...)
    surro : array_like
        Array of surrogates of shape (n_perm, n_amp, n_pha, ...)
    """
    s_mean, s_std = np.mean(surro, axis=0), np.std(surro, axis=0)
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
