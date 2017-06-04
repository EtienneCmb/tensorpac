"""Circular stastistics tools.

Code taken from the Circular Statistics Toolbox for Matlab
By Philipp Berens, 2009
Python adaptation by Etienne Combrisson
"""
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import chi2

__all__ = ['circ_corrcc', 'circ_r', 'circ_rtest']


def circ_corrcc(alpha, x):
    """Correlation coefficient between a circular and a linear random variable.

    Args:
        alpha: vector
            Sample of angles in radians

        x: vector
            Sample of linear random variable

    Returns:
        rho: float
            Correlation coefficient

        pval: float
            p-value
    """
    if len(alpha) is not len(x):
        raise ValueError('The length of alpha and x must be the same')
    n = len(alpha)

    # Compute correlation coefficent for sin and cos independently
    rxs = pearsonr(x, np.sin(alpha))[0]
    rxc = pearsonr(x, np.cos(alpha))[0]
    rcs = pearsonr(np.sin(alpha), np.cos(alpha))[0]

    # Compute angular-linear correlation (equ. 27.47)
    rho = np.sqrt((rxc**2 + rxs**2 - 2*rxc*rxs*rcs)/(1-rcs**2))

    # Compute pvalue
    pval = 1 - chi2.cdf(n*rho**2, 2)

    return rho, pval


def circ_r(alpha, w=None, d=0, axis=0):
    """Computes mean resultant vector length for circular data.

    Args:
        alpha: array
            Sample of angles in radians

    Kargs:
        w: array, optional, [def: None]
            Number of incidences in case of binned angle data

        d: radians, optional, [def: 0]
            Spacing of bin centers for binned data, if supplied
            correction factor is used to correct for bias in
            estimation of r

        axis: int, optional, [def: 0]
            Compute along this dimension

    Return:
        r: mean resultant length
    """
#     alpha = np.array(alpha)
#     if alpha.ndim == 1:
#         alpha = np.matrix(alpha)
#         if alpha.shape[0] is not 1:
#             alpha = alpha

    if w is None:
        w = np.ones(alpha.shape)
    elif (alpha.size is not w.size):
        raise ValueError("Input dimensions do not match")

    # Compute weighted sum of cos and sin of angles:
    r = np.multiply(w, np.exp(1j*alpha)).sum(axis=axis)

    # Obtain length:
    r = np.abs(r)/w.sum(axis=axis)

    # For data with known spacing, apply correction factor to
    # correct for bias in the estimation of r
    if d is not 0:
        c = d/2/np.sin(d/2)
        r = c*r

    return np.array(r)


def circ_rtest(alpha, w=None, d=0):
    """Computes Rayleigh test for non-uniformity of circular data.

    H0: the population is uniformly distributed around the circle
    HA: the populatoin is not distributed uniformly around the circle
    Assumption: the distribution has maximally one mode and the data is
    sampled from a von Mises distribution!

    Args:
        alpha: array
            Sample of angles in radians

    Kargs:
        w: array, optional, [def: None]
            Number of incidences in case of binned angle data

        d: radians, optional, [def: 0]
            Spacing of bin centers for binned data, if supplied
            correction factor is used to correct for bias in
            estimation of r.
    """
    alpha = np.array(alpha)
    if alpha.ndim == 1:
        alpha = np.matrix(alpha)
    if alpha.shape[1] > alpha.shape[0]:
        alpha = alpha.T

    if w is None:
        r = circ_r(alpha)
        n = len(alpha)
    else:
        if len(alpha) is not len(w):
            raise ValueError("Input dimensions do not match")
        r = circ_r(alpha, w, d)
        n = w.sum()

    # Compute Rayleigh's
    R = n*r
    z = (R**2) / n

    # Compute p value using approxation in Zar, p. 617
    pval = np.exp(np.sqrt(1+4*n+4*(n**2-R**2))-(1+2*n))

    return np.squeeze(pval), np.squeeze(z)
