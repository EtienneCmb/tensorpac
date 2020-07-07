"""Individual methods for assessing ERPAC."""
import numpy as np
from scipy.stats import chi2

from joblib import Parallel, delayed

from tensorpac.gcmi import nd_mi_gg
from tensorpac.config import CONFIG


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
    """Event-Related Phase Amplitude Coupling.

    This function computed the correlation coefficient between a circular and a
    linear random variable at each time point and across trials. Adapted from
    the function circ_corrcc Circular Statistics Toolbox for
    Matlab By Philipp Berens, 2009 :cite:`berens2009circstat`. This function is
    an adaptation of Voytek, 2013 :cite:`voytek2013method` for tensors.

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
    Voytek et al. 2013 :cite:`voytek2013method`
    """
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


def ergcpac(pha, amp, smooth=None, n_jobs=-1):
    """Event Related PAC using the Gaussian Copula Mutual Information.

    This function assumes that phases and amplitudes have already been
    prepared i.e. phases should be represented in a unit circle
    (np.c_[np.sin(pha), np.cos(pha)]) and both inputs should also have been
    copnormed.

    Parameters
    ----------
    pha, amp : array_like
        Respectively arrays of phases of shape (n_pha, n_times, 2, n_epochs)
        and the array of amplitudes of shape (n_amp, n_times, 1, n_epochs).

    Returns
    -------
    erpac : array_like
        Array of correlation coefficients of shape (n_amp, n_pha, n_times)

    References
    ----------
    Ince et al. 2017 :cite:`ince2017statistical`
    """
    # get shapes
    (n_pha, n_times, _, n_epochs), n_amp = pha.shape, amp.shape[0]  # noqa
    # compute mutual information across trials
    ergcpac = np.zeros((n_amp, n_pha, n_times))
    if isinstance(smooth, int):
        # define the temporal smoothing vector
        vec = np.arange(smooth, n_times - smooth, 1)
        times = [slice(k - smooth, k + smooth + 1) for k in vec]
        # move time axis to avoid to do it inside parallel
        pha, amp = np.moveaxis(pha, 1, -2), np.moveaxis(amp, 1, -2)
        # function to run in parallel across times
        def _fcn(t):  # noqa
            _erpac = np.zeros((n_amp, n_pha), dtype=float)
            xp, xa = pha[..., t, :], amp[..., t, :]
            for a in range(n_amp):
                _xa = xa.reshape(n_amp, 1, -1)
                for p in range(n_pha):
                    _xp = xp.reshape(n_pha, 2, -1)
                    _erpac[a, p] = nd_mi_gg(_xp[p, ...], _xa[a, ...])
            return _erpac
        # run the function across time points
        _ergcpac = Parallel(n_jobs=n_jobs, **CONFIG['JOBLIB_CFG'])(delayed(
            _fcn)(t) for t in times)
        # reconstruct the smoothed array
        for a in range(n_amp):
            for p in range(n_pha):
                mean_vec = np.zeros((n_times,), dtype=float)
                for t, _gc in zip(times, _ergcpac):
                    ergcpac[a, p, t] += _gc[a, p]
                    mean_vec[t] += 1
                ergcpac[a, p, :] /= mean_vec
    else:
        for a in range(n_amp):
            for p in range(n_pha):
                ergcpac[a, p, ...] = nd_mi_gg(pha[p, ...], amp[a, ...])
    return ergcpac


def swap_erpac_trials(pha):
    """Swap trials across the last dimension."""
    tr_ = np.random.permutation(pha.shape[-1])
    return pha[..., tr_]


def _ergcpac_perm(pha, amp, smooth=None, n_jobs=-1, n_perm=200):
    def _ergcpac_single_perm():
        p = swap_erpac_trials(pha)
        return ergcpac(p, amp, smooth=smooth, n_jobs=1)
    out = Parallel(n_jobs=n_jobs, **CONFIG['JOBLIB_CFG'])(delayed(
        _ergcpac_single_perm)() for _ in range(n_perm))
    return np.stack(out)
