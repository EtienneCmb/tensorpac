import numpy as np
from scipy.stats import chi2


def pear(x, y, st='i...j, k...j->ik...'):
    """Pearson correlation for PAC data.
    """
    # print('XY : ', x.shape, y.shape)
    n = x.shape[-1]
    # Distribution center :
    mu_x = x.mean(-1, keepdims=True)
    mu_y = y.mean(-1, keepdims=True)
    # Distribution deviation :
    s_x = x.std(-1, ddof=n-1, keepdims=True)
    s_y = y.std(-1, ddof=n-1, keepdims=True)
    # Compute correlation coefficient :
    cov = np.einsum(st, x, y)
    mu_xy = np.einsum(st, mu_x, mu_y)
    cov -= n * mu_xy
    cov /= np.einsum(st, s_x, s_y)
    return cov


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
    n = len(alpha)
    # Compute correlation coefficent for sin and cos independently
    sa, ca = np.sin(alpha), np.cos(alpha)
    rxs = pear(x, sa)
    rxc = pear(x, ca)
    rcs = pear(sa, ca, st='i...j, k...j->i...')[np.newaxis, ...]

    # Compute angular-linear correlation (equ. 27.47)
    rho = np.sqrt((rxc**2 + rxs**2 - 2*rxc*rxs*rcs)/(1-rcs**2))

    # Compute pvalue
    # pval = 1 - chi2.cdf(n*rho**2, 2)

    return rho#, pval
