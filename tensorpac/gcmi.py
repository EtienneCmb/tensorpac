"""Multi-dimentional Gaussian copula mutual information estimation."""
import numpy as np
from scipy.special import psi, ndtri


def ctransform(x):
    """Copula transformation (empirical CDF).

    Parameters
    ----------
    x : array_like
        Array of data. The trial axis should be the last one

    Returns
    -------
    xr : array_like
        Empirical CDF value along the last axis of x. Data is ranked and scaled
        within [0 1] (open interval)
    """
    xi = np.argsort(x)
    xr = np.argsort(xi).astype(float)
    xr += 1.
    xr /= float(xr.shape[-1] + 1)
    return xr


def copnorm(x):
    """Copula normalization.

    Parameters
    ----------
    x : array_like
        Array of data. The trial axis should be the last one

    Returns
    -------
    cx : array_like
        Standard normal samples with the same empirical CDF value as the input.
        Operates along the last axis
    """
    cx = ndtri(ctransform(x))
    # cx = sp.stats.norm.ppf(transform)
    return cx


def nd_mi_gg(x, y, mvaxis=None, traxis=-1, biascorrect=True, demeaned=False):
    """Multi-dimentional MI between two Gaussian variables in bits.

    Parameters
    ----------
    x, y : array_like
        Arrays to consider for computing the Mutual Information. The two input
        variables x and y should have the same shape except on the mvaxis
        (if needed).
    mvaxis : int | None
        Spatial location of the axis to consider if multi-variate analysis
        is needed
    traxis : int | -1
        Spatial location of the trial axis. By default the last axis is
        considered
    biascorrect : bool | True
        Specifies whether bias correction should be applied to the estimated MI
    demeaned : bool | False
        Specifies whether the input data already has zero mean (true if it has
        been copula-normalized)

    Returns
    -------
    mi : array_like
        The mutual information with the same shape as x and y, without the
        mvaxis and traxis
    """
    # Multi-dimentional shape checking
    # x.shape (..., x_mvaxis, traxis)
    # y.shape (..., y_mvaxis, traxis)
    ntrl = x.shape[-1]
    nvarx, nvary = x.shape[-2], y.shape[-2]
    nvarxy = nvarx + nvary

    # joint variable along the mvaxis
    xy = np.concatenate((x, y), axis=-2)
    if not demeaned:
        xy -= xy.mean(axis=-1, keepdims=True)
    cxy = np.einsum('...ij, ...kj->...ik', xy, xy)
    cxy /= float(ntrl - 1.)

    # submatrices of joint covariance
    cx = cxy[..., :nvarx, :nvarx]
    cy = cxy[..., nvarx:, nvarx:]

    # Cholesky decomposition
    chcxy = np.linalg.cholesky(cxy)
    chcx = np.linalg.cholesky(cx)
    chcy = np.linalg.cholesky(cy)

    # entropies in nats
    # normalizations cancel for mutual information
    hx = np.log(np.einsum('...ii->...i', chcx)).sum(-1)
    hy = np.log(np.einsum('...ii->...i', chcy)).sum(-1)
    hxy = np.log(np.einsum('...ii->...i', chcxy)).sum(-1)

    ln2 = np.log(2)
    if biascorrect:
        vec = np.arange(1, nvarxy + 1)
        psiterms = psi((ntrl - vec).astype(np.float) / 2.0) / 2.0
        dterm = (ln2 - np.log(ntrl - 1.0)) / 2.0
        hx = hx - nvarx * dterm - psiterms[:nvarx].sum()
        hy = hy - nvary * dterm - psiterms[:nvary].sum()
        hxy = hxy - nvarxy * dterm - psiterms[:nvarxy].sum()

    # MI in bits
    i = (hx + hy - hxy) / ln2
    return i
