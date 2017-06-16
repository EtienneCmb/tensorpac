"""Utility PAC functions.

- PacSignals : generate artificially phase-amplitude coupled signals
- PacVec : generate cross-frequency coupling vectors
"""
import numpy as np

__all__ = ['PacSignals', 'PacVec', 'PacPlot']


###############################################################################
###############################################################################
#                             SIGNALS
###############################################################################
###############################################################################


def PacSignals(fpha=2, famp=100, sf=1024, ndatasets=10, tmax=1, chi=0, noise=1,
               dpha=0, damp=0):
    """Generate artificially phase-amplitude coupled signals.

    Kargs:
        fpha: int/float, optional, [def: 2]
            Frequency for phase

        famp: int/float, optional, [def: 100]
            Frequency for amplitude

        sf: int, optional, [def: 1024]
            Sampling frequency

        ndatasets : int, optional, [def: 10]
            Number of datasets

        tmax: int/float (1<=tmax<=3), optional, [def: 1]
            Length of the time vector. If tmax=2 and sf=1024,
            the number of time points npts=1024*2=2048

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
    if (tmax < 1) or (tmax > 3):
        tmax = 1
    if (chi < 0) or (chi > 1):
        chi = 0
    if (noise < 0) or (noise > 3):
        noise = 0
    if (dpha < 0) or (dpha > 100):
        dpha = 0
    if (damp < 0) or (damp > 100):
        damp = 0
    fpha, famp = np.array(fpha), np.array(famp)
    time = np.arange(0, tmax, 1/sf)

    # Delta parameters :
    aPha = [fpha*(1-dpha/100), fpha*(1+dpha/100)]
    deltaPha = aPha[0] + (aPha[1]-aPha[0])*np.random.rand(ndatasets, 1)
    aAmp = [famp*(1-damp/100), famp*(1+damp/100)]
    deltaAmp = aAmp[0] + (aAmp[1]-aAmp[0])*np.random.rand(ndatasets, 1)

    # Generate the rnd datasets :
    data = np.zeros((ndatasets, len(time)))
    for k in range(ndatasets):
        # Create signals :
        xl = np.sin(2*np.pi*deltaPha[k]*time)
        xh = np.sin(2*np.pi*deltaAmp[k]*time)
        e = noise*np.random.rand(len(xl))

        # Create the coupling :
        ah = 0.5*((1 - chi) * xl + 1 + chi)
        al = 1
        data[k, :] = (ah*xh) + (al*xl) + e

    return data, time

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


###############################################################################
###############################################################################
#                             PLOTTING
###############################################################################
###############################################################################

def PacPlot():
    """Plot Pac."""
    import matplotlib.pyplot as plt
