"""Statistic tools."""
import numpy as np
from scipy.stats import chi2


def pearson(x, y, st='i...j, k...j->ik...', optimize=True):
    """Pearson correlation for multi-dimensional arrays.

    Args:
        x, y: np.ndarray
            Compute pearson correlation between the multi-dimensional arrays
            x and y.

    Kargs:
        st: string, optional, (def: 'i..j, k..j->ik...')
            The string to pass to the np.einsum function.

        optimize: bool, optional, (def: True)
            Optimize argument of the np.einsum function. Use either False,
            True, 'greedy' or 'optimal'.

    Returns:
        cov: np.ndarray
            The pearson correlation array.
    """
    n = x.shape[-1]
    # Distribution center :
    mu_x = x.mean(-1, keepdims=True)
    mu_y = y.mean(-1, keepdims=True)
    # Distribution deviation :
    s_x = x.std(-1, ddof=n-1, keepdims=True)
    s_y = y.std(-1, ddof=n-1, keepdims=True)
    # Compute correlation coefficient :
    cov = np.einsum(st, x, y, optimize=optimize)
    mu_xy = np.einsum(st, mu_x, mu_y, optimize=optimize)
    cov -= n * mu_xy
    cov /= np.einsum(st, s_x, s_y, optimize=optimize)
    return cov


def circ_corrcc(alpha, x, optimize=True):
    """Correlation coefficient between a circular and a linear random variable.

    Code from the Circular Statistics Toolbox for Matlab By Philipp Berens 2009
    and adapted for multi-dimensional arrays.

    Args:
        alpha: vector
            Sample of angles in radians

        x: vector
            Sample of linear random variable

        optimize: bool, optional, (def: True)
            Optimize argument of the np.einsum function. Use either False,
            True, 'greedy' or 'optimal'.

    Returns:
        rho: float
            Correlation coefficient.

        pval: float
            P-value.
    """
    n = alpha.shape[-1]
    # Compute correlation coefficient for sin and cos independently
    sa, ca = np.sin(alpha), np.cos(alpha)
    rxs = pearson(x, sa, optimize=optimize)
    rxc = pearson(x, ca, optimize=optimize)
    rcs = pearson(sa, ca, st='i...j, k...j->i...', optimize=optimize)
    rcs = rcs[np.newaxis, ...]

    # Compute angular-linear correlation (equ. 27.47)
    rho = np.sqrt((rxc**2 + rxs**2 - 2*rxc*rxs*rcs)/(1-rcs**2))

    # Compute pvalue :
    pval = 1. - chi2.cdf(n*rho**2, 2)

    return rho, pval
