"""Extract spectral informations from data."""
import numpy as np
from joblib import Parallel, delayed
from scipy.signal import hilbert
from .filtering import filtdata

__all__ = ['spectral']


def spectral(x, sf, f, axis, stype, dcomplex, filt, filtorder, cycle, width,
             njobs):
    """Extract spectral informations from data.

    Args:
        x: np.ndarray
            Array of data

        sf: float
            Sampling frequency

        f: np.ndarray
            Frequency vector of shape (N, 2)

        axis: int
            Axis where the time is located.

        stype: string
            Spectral informations to extract (use either 'pha' or 'amp')

        dcomplex: string
            Complex decomposition type. Use either 'hilbert' or 'wavelet'

        filt: string
            Name of the filter to use (only if dcomplex is 'hilbert'). Use
            either 'eegfilt', 'butter' or 'bessel'.

        filtorder: int
            Order of the filter (only if dcomplex is 'hilbert')

        cycle: int
            Number of cycles to use for fir1 filtering.

        width: int
            Width of the wavelet.

        njobs: int
            Number of jobs to use. If jobs is -1, all of them are going to be
            used.
    """
    # Filtering + complex decomposition :
    if dcomplex is 'hilbert':
        # Filt each time series :
        nf = range(f.shape[0])
        xf = Parallel(n_jobs=njobs)(delayed(filtdata)(
                  x, sf, f[k, :], axis, filt, cycle, filtorder) for k in nf)
        # Use hilbert for the complex decomposition :
        xd = hilbert(xf, axis=axis+1) if stype is not None else np.array(xf)
    elif dcomplex is 'wavelet':
        f = f.mean(1)  # centered frequencies
        xd = Parallel(n_jobs=njobs)(delayed(morlet)(
                                            x, sf, k, axis, width) for k in f)

    # Extract phase / amplitude :
    if stype is 'pha':
        return np.angle(np.moveaxis(xd, axis+1, -1))
    elif stype is 'amp':
        return np.abs(np.moveaxis(xd, axis+1, -1))
    elif stype is None:
        return np.moveaxis(xd, axis+1, -1)


def morlet(x, sf, f, axis=0, width=7.):
    """Complex decomposition of a signal x using the morlet wavelet.

    Args:
        x: np.ndarray, shape (N,)
            The signal to use for the complex decomposition. Must be
            a vector of length N.

        sf: float
            Sampling frequency

        f: np.ndarray, shape (2,)
            Frequency vector

    Kargs:
        width: float, optional, (def: 7.)
            Width of the wavelet

        axis: int, optional, (def: 0)
            Axis along performing the convolution.

    Returns:
        xout: np.ndarray, shape (N,)
            The complex decomposition of the signal x.
    """
    dt = 1 / sf
    sf = f / width
    st = 1 / (2 * np.pi * sf)

    # Build morlet's wavelet :
    t = np.arange(-width * st / 2, width * st / 2, dt)
    A = 1 / np.sqrt((st * np.sqrt(np.pi)))
    m = A * np.exp(-np.square(t) / (2 * np.square(st))) * np.exp(
                                                       1j * 2 * np.pi * f * t)

    def ndmorlet(xt):
        # Compute morlet :
        y = np.convolve(xt, m)
        return y[int(np.ceil(len(m) / 2)) - 1:int(len(y) - np.floor(
                                                                len(m) / 2))]
    return np.apply_along_axis(ndmorlet, axis, x)
