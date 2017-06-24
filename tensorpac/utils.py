"""Utility PAC functions.

- PacSignals : generate artificially phase-amplitude coupled signals
- PacVec : generate cross-frequency coupling vectors
"""
import numpy as np

__all__ = ['PacSignals', 'PacVec', 'PacTriVec']


###############################################################################
###############################################################################
#                             SIGNALS
###############################################################################
###############################################################################

def PacSignals(fpha=10, famp=100, sf=1024, npts=4000, ndatasets=10, chi=0,
               noise=1, dpha=0, damp=0):
    """Generate artificially phase-amplitude coupled signals.

    Kargs:
        fpha: int/float, optional, [def: 10]
            Frequency for phase

        famp: int/float, optional, [def: 100]
            Frequency for amplitude

        sf: int, optional, [def: 1024]
            Sampling frequency

        ndatasets : int, optional, [def: 10]
            Number of datasets

        npts: int, optional, [def: 4000]
            Number of points for each signal.

        chi: int/float (0<=chi<=1), optional, [def: 0]
            Amount of coupling. If chi=0, signals of phase and amplitude
            are strongly coupled.

        noise: int/float (0<=noise<=3), optional, [def: 1]
            Amount of noise

        dpha: int/float (0<=dpha<=100), optional, [def: 0]
            Introduce a random incertitude on the phase frequency.
            If fpha is 2, and dpha is 50, the frequency for the phase signal
            will be between :
            [2-0.5*2, 2+0.5*2]=[1,3]

        damp: int/float (0<=damp<=100), optional, [def: 0]
            Introduce a random incertitude on the amplitude frequency.
            If famp is 60, and damp is 10, the frequency for the amplitude
            signal will be between :
            [60-0.1*60, 60+0.1*60]=[54,66]

    Return:
        data: array
            The randomly coupled signals. The shape of data will be
            (ndatasets x npts)

        time: array
            The corresponding time vector
    """
    # Check the inputs variables :
    if not 0 <= chi <= 1:
        chi = 0
    if not 0 <= noise <= 3:
        noise = 0
    if not 0 <= dpha <= 100:
        dpha = 0
    if not 0 <= damp <= 100:
        damp = 0
    fpha, famp = np.asarray(fpha), np.asarray(famp)
    time = np.mgrid[0:ndatasets, 0:npts][1] / sf
    data = np.zeros_like(time)

    # Band / Delta parameters :
    if fpha.ndim == 0:
        aPha = [fpha*(1-dpha/100), fpha*(1+dpha/100)]
        deltaPha = aPha[0] + (aPha[1]-aPha[0])*np.random.rand(ndatasets, 1)
    elif fpha.ndim == 1:
        deltaPha = np.random.uniform(fpha[0], fpha[1], ndatasets)
    if famp.ndim == 0:
        aAmp = [famp*(1-damp/100), famp*(1+damp/100)]
        deltaAmp = aAmp[0] + (aAmp[1]-aAmp[0])*np.random.rand(ndatasets, 1)
    elif famp.ndim == 1:
        deltaAmp = np.random.uniform(famp[0], famp[1], ndatasets)

    # Reshape phase/amplitude bands :
    deltaPha, deltaAmp = deltaPha.reshape(-1, 1), deltaAmp.reshape(-1, 1)

    # Create phase and amplitude signals :
    xl = np.sin(2 * np.pi * deltaPha * time)
    xh = np.sin(2 * np.pi * deltaAmp * time)

    # Create the coupling :
    ah = .5 * ((1. - chi) * xl + 1. + chi)
    al = 1.

    # Generate datasets :
    data = (ah * xh) + (al * xl)
    data += noise * np.random.rand(*data.shape)  # Add noise

    return data, time[0, :]

###############################################################################
###############################################################################
#                             FREQUENCY VECTOR
###############################################################################
###############################################################################


def PacVec(fpha=(2, 30, 2, 1), famp=(60, 200, 10, 5)):
    """Generate cross-frequency coupling vectors.

    Kargs:
        fpha: tuple, optional, [def: (2, 30, 2, 1)]
            Frequency parameters for phase. Each argument inside the tuple
            mean (starting fcy, ending fcy, width, step)

        famp: tuple, optional, [def: (60, 200, 10, 5)]
            Frequency parameters for amplitude. Each argument inside the
            tuple mean (starting fcy, ending fcy, width, step)

    Returns:
        pVec: np.ndarray, shape (N, 2)
            Array containing the pairs of phase frequencies.

        aVec: np.ndarray, shape (N, 2)
            Array containing the pairs of amplitude frequencies.
    """
    return _CheckFreq(fpha), _CheckFreq(famp)


def _CheckFreq(f):
    """Check the frequency definition."""
    f = np.atleast_2d(np.asarray(f))
    #
    if len(f.reshape(-1)) == 1:
        raise ValueError("The length of f should at least be 2.")
    elif 2 in f.shape:  # f of shape (N, 2) or (2, N)
        if f.shape[1] is not 2:
            f = f.T
    elif np.squeeze(f).shape == (4,):  # (fstart, fend, fwidth, fstep)
        f = _CreatePairsVector(*tuple(np.squeeze(f)))
    else:  # Sequential
        f = f.reshape(-1)
        f.sort()
        f = np.c_[f[0:-1], f[1::]]
    return f


def _CreatePairsVector(fstart, fend, fwidth, fstep):
    # Generate two array for phase and amplitude :
    fdown = np.arange(fstart, fend-fwidth, fstep)
    fup = np.arange(fstart+fwidth, fend, fstep)
    return np.c_[fdown, fup]


def PacTriVec(fstart=60, fend=160, fwidth=10):
    """Generate triangular vector.

    Kargs:
        fstart: float, optional, (def: 60)
            Starting frequency.

        fend: float, optional, (def: 160)
            Ending frequency.

        fwidth: float, optional, (def: 10)
            Frequency bandwidth.

    Returns:
        f: np.ndarray
            The triangular vector.

        tridx: np.ndarray
            The triangular index for the reconstruction.
    """
    starting = np.arange(fstart, fend+fwidth, fwidth)
    f, tridx = np.array([]), np.array([])
    for num, k in enumerate(starting[0:-1]):
        # Lentgh of the vector to build :
        L = len(starting) - (num + 1)
        # Create the frequency vector for this starting frequency :
        fst = np.c_[np.full(L, k), starting[num+1::]]
        nfst = fst.shape[0]
        # Create the triangular index for this vector of frequencies :
        idx = np.c_[np.flipud(np.arange(nfst)), np.full(nfst, num)]
        tridx = np.concatenate((tridx, idx), axis=0) if tridx.size else idx
        f = np.concatenate((f, fst), axis=0) if f.size else fst
    return f, tridx
