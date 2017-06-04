import numpy as np
from scipy.signal import filtfilt, butter, bessel


__all__ = ['filtdata']

###############################################################################
###############################################################################
#                                FILT DATA
###############################################################################
###############################################################################


def filtdata(x, sf, f, axis, filt, cycle, filtorder):
    """Filt the data using a forward/backward filter to avoid phase shifting.

    Args:
        x: np.ndarray
            Array of data

        sf: float
            Sampling frequency

        f: np.ndarray
            Frequency vector of shape (N, 2)

        axis: int
            Axis where the time is located.

        filt: string
            Name of the filter to use (only if dcomplex is 'hilbert'). Use
            either 'eegfilt', 'butter' or 'bessel'.

        filtorder: int
            Order of the filter (only if dcomplex is 'hilbert')

        cycle: int
            Number of cycles to use for fir1 filtering.
    """
    # fir1 filter :
    if filt == 'fir1':
        fOrder = fir_order(sf, x.shape[axis], f[0], cycle=cycle)
        b, a = fir1(fOrder, f/(sf / 2))

    # butterworth filter :
    elif filt == 'butter':
        b, a = butter(filtorder, [(2*f[0])/sf, (2*f[1])/sf], btype='bandpass')
        fOrder = None

    # bessel filter :
    elif filt == 'bessel':
        b, a = bessel(filtorder, [(2*f[0])/sf, (2*f[1])/sf], btype='bandpass')
        fOrder = None

    return filtfilt(b, a, x, padlen=fOrder, axis=axis)

###############################################################################
###############################################################################
#                       FILTER ORDER
###############################################################################
###############################################################################


def fir_order(Fs, sizevec, flow, cycle=3):
    filtorder = cycle * (Fs // flow)

    if (sizevec < 3 * filtorder):
        filtorder = (sizevec - 1) // 3

    return int(filtorder)


###############################################################################
###############################################################################
#                            FIR1
###############################################################################
###############################################################################


def NoddFcn(F, M, W, L):
    """Odd case."""
    # Variables :
    b0 = 0
    m = np.array(range(int(L + 1)))
    k = m[1:len(m)]
    b = np.zeros(k.shape)

    # Run Loop :
    for s in range(0, len(F), 2):
        m = (M[s + 1] - M[s]) / (F[s + 1] - F[s])
        b1 = M[s] - m * F[s]
        b0 = b0 + (b1 * (F[s + 1] - F[s]) + m / 2 * (
            F[s + 1] * F[s + 1] - F[s] * F[s])) * abs(
            np.square(W[round((s + 1) / 2)]))
        b = b + (m / (4 * np.pi * np.pi) * (
            np.cos(2 * np.pi * k * F[s + 1]) - np.cos(2 * np.pi * k * F[s])
        ) / (k * k)) * abs(np.square(W[round((s + 1) / 2)]))
        b = b + (F[s + 1] * (m * F[s + 1] + b1) * np.sinc(2 * k * F[
          s + 1]) - F[s] * (m * F[s] + b1) * np.sinc(2 * k * F[s])) * abs(
            np.square(W[round((s + 1) / 2)]))

    b = np.insert(b, 0, b0)
    a = (np.square(W[0])) * 4 * b
    a[0] = a[0] / 2
    aud = np.flipud(a[1:len(a)]) / 2
    a2 = np.insert(aud, len(aud), a[0])
    h = np.concatenate((a2, a[1:] / 2))

    return h


def NevenFcn(F, M, W, L):
    """Even case."""
    # Variables :
    k = np.array(range(0, int(L) + 1, 1)) + 0.5
    b = np.zeros(k.shape)

    # # Run Loop :
    for s in range(0, len(F), 2):
        m = (M[s + 1] - M[s]) / (F[s + 1] - F[s])
        b1 = M[s] - m * F[s]
        b = b + (m / (4 * np.pi * np.pi) * (np.cos(2 * np.pi * k * F[
            s + 1]) - np.cos(2 * np.pi * k * F[s])) / (
            k * k)) * abs(np.square(W[round((s + 1) / 2)]))
        b = b + (F[s + 1] * (m * F[s + 1] + b1) * np.sinc(2 * k * F[
          s + 1]) - F[s] * (m * F[s] + b1) * np.sinc(2 * k * F[s])) * abs(
            np.square(W[round((s + 1) / 2)]))

    a = (np.square(W[0])) * 4 * b
    h = 0.5 * np.concatenate((np.flipud(a), a))

    return h


def firls(N, F, M):
    # Variables definition :
    W = np.ones(round(len(F) / 2))
    N += 1
    F /= 2
    L = (N - 1) / 2

    Nodd = bool(N % 2)

    if Nodd:  # Odd case
        h = NoddFcn(F, M, W, L)
    else:  # Even case
        h = NevenFcn(F, M, W, L)

    return h


####################################################################
# - Compute the window :
####################################################################
def fir1(N, Wn):
    # Variables definition :
    nbands = len(Wn) + 1
    ff = np.array((0, Wn[0], Wn[0], Wn[1], Wn[1], 1))

    f0 = np.mean(ff[2:4])
    L = N + 1

    mags = np.array(range(nbands)) % 2
    aa = np.ravel(np.matlib.repmat(mags, 2, 1), order='F')

    # Get filter coefficients :
    h = firls(L - 1, ff, aa)

    # Apply a window to coefficients :
    Wind = np.hamming(L)
    b = np.matrix(h.T * Wind)
    c = np.matrix(np.exp(-1j * 2 * np.pi * (f0 / 2) * np.array(range(L))))
    b = b / abs(c * b.T)

    return np.squeeze(np.array(b)), 1
