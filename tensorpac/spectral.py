"""Extract spectral informations from data."""
import numpy as np
from joblib import Parallel, delayed
from scipy.signal import hilbert
from scipy import fftpack

from tensorpac.filtering import filtdata
from tensorpac.config import JOBLIB_CFG


def hilbertm(x):
    """Faster Hilbert fix.

    x must have a shape of (..., n_pts)
    """
    n_pts = x.shape[-1]
    fc = fftpack.helper.next_fast_len(n_pts)
    return hilbert(x, fc, axis=-1)[..., 0:n_pts]


def spectral(x, sf, f, stype, dcomplex, filt, filtorder, cycle, width,
             n_jobs):
    """Extract spectral informations from data.

    Parameters
    ----------
    x : array_like
        Array of data
    sf : float
        Sampling frequency
    f : array_like
        Frequency vector of shape (N, 2)
    stype : string
        Spectral informations to extract (use either 'pha' or 'amp')
    dcomplex : string
        Complex decomposition type. Use either 'hilbert' or 'wavelet'
    filt : string
        Name of the filter to use (only if dcomplex is 'hilbert'). Use
        either 'eegfilt', 'butter' or 'bessel'.
    filtorder : int
        Order of the filter (only if dcomplex is 'hilbert')
    cycle : int
        Number of cycles to use for fir1 filtering.
    width : int
        Width of the wavelet.
    n_jobs : int
        Number of jobs to use. If jobs is -1, all of them are going to be
        used.
    """
    # Filtering + complex decomposition :
    if dcomplex is 'hilbert':
        # Filt each time series :
        nf = range(f.shape[0])
        xf = Parallel(n_jobs=n_jobs, **JOBLIB_CFG)(delayed(filtdata)(
            x, sf, f[k, :], filt, cycle, filtorder) for k in nf)
        # Use hilbert for the complex decomposition :
        xf = np.asarray(xf)
        if stype is not None:
            xd = hilbertm(xf)
    elif dcomplex is 'wavelet':
        f = f.mean(1)  # centered frequencies
        xd = Parallel(n_jobs=n_jobs, **JOBLIB_CFG)(delayed(morlet)(
            x, sf, k, width) for k in f)

    # Extract phase / amplitude :
    if stype is 'pha':
        return np.angle(xd)
    elif stype is 'amp':
        return np.abs(xd)
    elif stype is None:
        return xd


def morlet(x, sf, f, width=7.):
    """Complex decomposition of a signal x using the morlet wavelet.

    Parameters
    ----------
    x : array_like, shape (N,)
        The signal to use for the complex decomposition. Must be
        a vector of length N.
    sf : float
        Sampling frequency
    f : array_like, shape (2,)
        Frequency vector
    width : float | 7.
        Width of the wavelet

    Returns
    -------
    xout: array_like, shape (N,)
        The complex decomposition of the signal x.
    """
    dt = 1 / sf
    sf = f / width
    st = 1 / (2 * np.pi * sf)

    # Build morlet's wavelet :
    t = np.arange(-width * st / 2, width * st / 2, dt)
    a = 1 / np.sqrt((st * np.sqrt(np.pi)))
    m = a * np.exp(-np.square(t) / (2 * np.square(st))) * np.exp(
        1j * 2 * np.pi * f * t)

    def ndmorlet(xt):
        # Compute morlet :
        y = np.convolve(xt, m)
        return y[int(np.ceil(len(m) / 2)) - 1:int(len(y) - np.floor(
            len(m) / 2))]
    return np.apply_along_axis(ndmorlet, -1, x)
