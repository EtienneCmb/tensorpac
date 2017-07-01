"""Utility PAC functions.

- pac_signals : generate artificially phase-amplitude coupled signals
- pac_vec : generate cross-frequency coupling vectors
"""
import numpy as np

__all__ = ('pac_signals', 'pac_vec', 'pac_trivec')


###############################################################################
###############################################################################
#                             SIGNALS
###############################################################################
###############################################################################

def pac_signals(fpha=10, famp=100, sf=1024, npts=4000, ndatasets=10, chi=0,
                noise=1., dpha=0, damp=0):
    """Generate artificially phase-amplitude coupled signals.

    Parameters
    ----------
    fpha : int/float, optional
        Frequency for phase
    famp : int/float, optional
        Frequency for amplitude
    sf : int, optional
        Sampling frequency
    ndatasets : int, optional
        Number of datasets
    npts : int, optional
        Number of points for each signal.
    chi : float, optional
        Amount of coupling. If chi=0, signals of phase and amplitude
        are strongly coupled (0.<=chi<=1.).
    noise : float, optional
        Amount of noise (0<=noise<=3).
    dpha : float, optional
        Random incertitude on phase frequences (0<=dpha<=100). If fpha is 2,
        and dpha is 50, the frequency for the phase signal will be between :
        [2-0.5*2, 2+0.5*2]=[1,3]
    damp : float, optional
        Random incertitude on amplitude frequencies (0<=damp<=100). If famp is
        60, and damp is 10, the frequency for the amplitude signal will be
        between : [60-0.1*60, 60+0.1*60]=[54,66]

    Returns
    -------
    data : array_like
        The randomly coupled signals of shape (ndatasets, npts).
    time : array_like
        The corresponding time vector according to the defined number of points
        and sampling frequency.
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
    sh = (ndatasets, 1)
    if fpha.ndim == 0:
        apha = [fpha * (1 - dpha / 100), fpha * (1 + dpha / 100)]
        del_pha = apha[0] + (apha[1] - apha[0]) * np.random.rand(*sh)
    elif fpha.ndim == 1:
        del_pha = np.random.uniform(fpha[0], fpha[1], ndatasets)
    if famp.ndim == 0:
        a_amp = [famp * (1 - damp / 100), famp * (1 + damp / 100)]
        del_amp = a_amp[0] + (a_amp[1] - a_amp[0]) * np.random.rand(*sh)
    elif famp.ndim == 1:
        del_amp = np.random.uniform(famp[0], famp[1], ndatasets)

    # Reshape phase/amplitude bands :
    del_pha, del_amp = del_pha.reshape(-1, 1), del_amp.reshape(-1, 1)

    # Create phase and amplitude signals :
    xl = np.sin(2 * np.pi * del_pha * time)
    xh = np.sin(2 * np.pi * del_amp * time)

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


def pac_vec(fpha=(2, 30, 2, 1), famp=(60, 200, 10, 5)):
    """Generate cross-frequency coupling vectors.

    Parameters
    ----------
    fpha, famp : tuple, optional
        Frequency parameters for phase and amplitude. Each argument inside the
        tuple mean (starting fcy, ending fcy, bandwidth, step).

    Returns
    -------
    pvec, avec : array_like
        Arrays containing the pairs of phase and amplitude frequencies. Each
        vector have a shape of (N, 2).
    """
    return _check_freq(fpha), _check_freq(famp)


def _check_freq(f):
    """Check the frequency definition."""
    f = np.atleast_2d(np.asarray(f))
    #
    if len(f.reshape(-1)) == 1:
        raise ValueError("The length of f should at least be 2.")
    elif 2 in f.shape:  # f of shape (N, 2) or (2, N)
        if f.shape[1] is not 2:
            f = f.T
    elif np.squeeze(f).shape == (4,):  # (fstart, fend, fwidth, fstep)
        f = _pair_vectors(*tuple(np.squeeze(f)))
    else:  # Sequential
        f = f.reshape(-1)
        f.sort()
        f = np.c_[f[0:-1], f[1::]]
    return f


def _pair_vectors(fstart, fend, fwidth, fstep):
    # Generate two array for phase and amplitude :
    fdown = np.arange(fstart, fend - fwidth, fstep)
    fup = np.arange(fstart + fwidth, fend, fstep)
    return np.c_[fdown, fup]


def pac_trivec(fstart=60., fend=160., fwidth=10.):
    """Generate triangular vector.

    By contrast with the pac_vec function, this function generate frequency
    vector with an increasing frequency bandwidth.

    Parameters
    ----------
    fstart : float, optional
        Starting frequency.
    fend : float, optional
        Ending frequency.
    fwidth : float, optional
        Frequency bandwidth increase between each band.

    Returns
    -------
    f : array_like
        The triangular vector.
    tridx : array_like
        The triangular index for the reconstruction.
    """
    starting = np.arange(fstart, fend + fwidth, fwidth)
    f, tridx = np.array([]), np.array([])
    for num, k in enumerate(starting[0:-1]):
        # Lentgh of the vector to build :
        l = len(starting) - (num + 1)
        # Create the frequency vector for this starting frequency :
        fst = np.c_[np.full(l, k), starting[num + 1::]]
        nfst = fst.shape[0]
        # Create the triangular index for this vector of frequencies :
        idx = np.c_[np.flipud(np.arange(nfst)), np.full(nfst, num)]
        tridx = np.concatenate((tridx, idx), axis=0) if tridx.size else idx
        f = np.concatenate((f, fst), axis=0) if f.size else fst
    return f, tridx
