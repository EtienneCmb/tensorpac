"""Extract spectral informations from data."""
import numpy as np
from joblib import Parallel, delayed
from scipy.signal import hilbert, filtfilt
from scipy import fftpack

from tensorpac.config import CONFIG


def hilbertm(x):
    """Faster Hilbert fix.

    x must have a shape of (..., n_pts)
    """
    n_pts = x.shape[-1]
    fc = fftpack.helper.next_fast_len(n_pts)
    return hilbert(x, fc, axis=-1)[..., 0:n_pts]


def spectral(x, sf, f, stype, dcomplex, cycle, width, n_jobs):
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
    cycle : int
        Number of cycles to use for fir1 filtering.
    width : int
        Width of the wavelet.
    n_jobs : int
        Number of jobs to use. If jobs is -1, all of them are going to be
        used.
    """
    n_freqs = f.shape[0]
    # Filtering + complex decomposition :
    if dcomplex is 'hilbert':
        # get filtering coefficients
        b = []
        a = np.zeros((n_freqs,), dtype=float)
        forder = np.zeros((n_freqs,), dtype=int)
        for k in range(n_freqs):
            forder[k] = fir_order(sf, x.shape[-1], f[k, 0], cycle=cycle)
            _b, a[k] = fir1(forder[k], f[k, :] / (sf / 2.))
            b += [_b]
        # Filt each time series :
        xf = Parallel(n_jobs=n_jobs, **CONFIG['JOBLIB_CFG'])(delayed(filtfilt)(
            b[k], a[k], x, padlen=forder[k], axis=-1) for k in range(n_freqs))
        # Use hilbert for the complex decomposition :
        xd = np.asarray(xf)
        if stype is not None:
            xd = hilbertm(xd)
    elif dcomplex is 'wavelet':
        f = f.mean(1)  # centered frequencies
        xd = Parallel(n_jobs=n_jobs, **CONFIG['JOBLIB_CFG'])(delayed(morlet)(
            x, sf, k, width) for k in f)

    # Extract phase / amplitude :
    if stype is 'pha':
        return np.angle(xd).astype(np.float64)
    elif stype is 'amp':
        return np.abs(xd).astype(np.float64)
    elif stype is None:
        return xd.astype(np.float64)

###############################################################################
###############################################################################
#                            FILTERING
###############################################################################
###############################################################################


def fir_order(fs, sizevec, flow, cycle=3):
    filtorder = cycle * (fs // flow)

    if (sizevec < 3 * filtorder):
        filtorder = (sizevec - 1) // 3

    return int(filtorder)


def n_odd_fcn(f, o, w, l):
    """Odd case."""
    # Variables :
    b0 = 0
    m = np.array(range(int(l + 1)))
    k = m[1:len(m)]
    b = np.zeros(k.shape)

    # Run Loop :
    for s in range(0, len(f), 2):
        m = (o[s + 1] - o[s]) / (f[s + 1] - f[s])
        b1 = o[s] - m * f[s]
        b0 = b0 + (b1 * (f[s + 1] - f[s]) + m / 2 * (
            f[s + 1] * f[s + 1] - f[s] * f[s])) * abs(
            np.square(w[round((s + 1) / 2)]))
        b = b + (m / (4 * np.pi * np.pi) * (
            np.cos(2 * np.pi * k * f[s + 1]) - np.cos(2 * np.pi * k * f[s])
        ) / (k * k)) * abs(np.square(w[round((s + 1) / 2)]))
        b = b + (f[s + 1] * (m * f[s + 1] + b1) * np.sinc(2 * k * f[
            s + 1]) - f[s] * (m * f[s] + b1) * np.sinc(2 * k * f[s])) * abs(
            np.square(w[round((s + 1) / 2)]))

    b = np.insert(b, 0, b0)
    a = (np.square(w[0])) * 4 * b
    a[0] = a[0] / 2
    aud = np.flipud(a[1:len(a)]) / 2
    a2 = np.insert(aud, len(aud), a[0])
    h = np.concatenate((a2, a[1:] / 2))

    return h


def n_even_fcn(f, o, w, l):
    """Even case."""
    # Variables :
    k = np.array(range(0, int(l) + 1, 1)) + 0.5
    b = np.zeros(k.shape)

    # # Run Loop :
    for s in range(0, len(f), 2):
        m = (o[s + 1] - o[s]) / (f[s + 1] - f[s])
        b1 = o[s] - m * f[s]
        b = b + (m / (4 * np.pi * np.pi) * (np.cos(2 * np.pi * k * f[
            s + 1]) - np.cos(2 * np.pi * k * f[s])) / (
            k * k)) * abs(np.square(w[round((s + 1) / 2)]))
        b = b + (f[s + 1] * (m * f[s + 1] + b1) * np.sinc(2 * k * f[
            s + 1]) - f[s] * (m * f[s] + b1) * np.sinc(2 * k * f[s])) * abs(
            np.square(w[round((s + 1) / 2)]))

    a = (np.square(w[0])) * 4 * b
    h = 0.5 * np.concatenate((np.flipud(a), a))

    return h


def firls(n, f, o):
    # Variables definition :
    w = np.ones(round(len(f) / 2))
    n += 1
    f /= 2
    lo = (n - 1) / 2

    nodd = bool(n % 2)

    if nodd:  # Odd case
        h = n_odd_fcn(f, o, w, lo)
    else:  # Even case
        h = n_even_fcn(f, o, w, lo)

    return h


def fir1(n, wn):
    # Variables definition :
    nbands = len(wn) + 1
    ff = np.array((0, wn[0], wn[0], wn[1], wn[1], 1))

    f0 = np.mean(ff[2:4])
    lo = n + 1

    mags = np.array(range(nbands)).reshape(1, -1) % 2
    aa = np.ravel(np.tile(mags, (2, 1)), order='F')

    # Get filter coefficients :
    h = firls(lo - 1, ff, aa)

    # Apply a window to coefficients :
    wind = np.hamming(lo)
    b = h * wind
    c = np.exp(-1j * 2 * np.pi * (f0 / 2) * np.array(range(lo)))
    b /= abs(c @ b)

    return b, 1


###############################################################################
###############################################################################
#                            FILTERING
###############################################################################
###############################################################################


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
