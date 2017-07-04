"""Utility PAC functions.

- pac_signals_tort : generate artificially phase-amplitude coupled signals
- pac_vec : generate cross-frequency coupling vectors
"""
import numpy as np
from .spectral import morlet

__all__ = ('pac_signals_wavelet', 'pac_signals_tort', 'pac_vec', 'pac_trivec')


###############################################################################
###############################################################################
#                             SIGNALS
###############################################################################
###############################################################################


def pac_signals_wavelet(fpha=10., famp=100., sf=1024., npts=4000., ntrials=10,
                        noise=.1, pp=0., rnd_state=0):
    """Generate artificially phase-amplitude coupled signals using wavelets.

    This function is inspired by the code of the pactools toolbox developped by
    Tom Dupre la Tour.

    Parameters
    ----------
    fpha : float | 10.
        Frequency for phase. Use either a float number for a centered frequency
        of a band (like [5, 7]) for a bandwidth.

    famp : float | 100.
        Frequency for amplitude. Use either a float number for a centered
        frequency of a band (like [60, 80]) for a bandwidth.

    sf : float | 1024.
        Sampling frequency.

    npts : int | 4000
        Number of time points.

    ntrials : int | 10
        Number of trials in the dataset.

    noise : float | .1
        Amount of white noise.

    pp : float | 0.
        The preferred-phase of the coupling.

    rnd_state: int | 0
        Fix random of the machine (for reproducibility)

    Returns
    -------
    data : array_like
        Array of pac signals of shape (ntrials, npts).

    time : array_like
        Time vector of shape (npts,).
    """
    npts = int(npts)
    sf = float(sf)
    fpha, famp = np.asarray(fpha).mean(), np.asarray(famp).mean()
    time = np.mgrid[0:ntrials, 0:npts][1] / sf
    # Random state of the machine :
    rng = np.random.RandomState(rnd_state)
    # Get complex decomposition of random points in the phase frequency band :
    driver = morlet(rng.randn(ntrials, npts), sf, fpha, axis=1)
    driver /= np.max(driver, axis=1, keepdims=True)
    # Create amplitude signals :
    xh = np.sin(2 * np.pi * famp * time)
    dpha = np.exp(-1j * pp)
    modulation = 1. / (1. + np.exp(- 6. * 1. * np.real(driver * dpha)))
    # Modulate the amplitude :
    xh *= modulation
    # Get the phase signal :
    xl = np.real(driver)
    # Build the pac signal :
    data = xh + xl + noise * rng.randn(*xh.shape)

    return data, time[0, :]


def pac_signals_tort(fpha=10., famp=100., sf=1024, npts=4000, ntrials=10,
                     chi=0., noise=1., dpha=0., damp=0., rnd_state=0):
    """Generate artificially phase-amplitude coupled signals.

    This function use the definition of Tort et al. 2010.

    Parameters
    ----------
    fpha : float | 10.
        Frequency for phase. Use either a float number for a centered frequency
        of a band (like [5, 7]) for a bandwidth.

    famp : float | 100.
        Frequency for amplitude. Use either a float number for a centered
        frequency of a band (like [60, 80]) for a bandwidth.

    sf : int | 1024
        Sampling frequency

    ntrials : int | 10
        Number of datasets

    npts : int | 4000
        Number of points for each signal.

    chi : float | 0.
        Amount of coupling. If chi=0, signals of phase and amplitude
        are strongly coupled (0.<=chi<=1.).

    noise : float | 1.
        Amount of noise (0<=noise<=3).

    dpha : float | 0.
        Random incertitude on phase frequences (0<=dpha<=100). If fpha is 2,
        and dpha is 50, the frequency for the phase signal will be between :
        [2-0.5*2, 2+0.5*2]=[1,3]

    damp : float | 0.
        Random incertitude on amplitude frequencies (0<=damp<=100). If famp is
        60, and damp is 10, the frequency for the amplitude signal will be
        between : [60-0.1*60, 60+0.1*60]=[54,66]

    rnd_state: int | 0
        Fix random of the machine (for reproducibility)

    Returns
    -------
    data : array_like
        Array of pac signals of shape (ntrials, npts).

    time : array_like
        Time vector of shape (npts,).
    """
    npts = int(npts)
    sf = float(sf)
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
    time = np.mgrid[0:ntrials, 0:npts][1] / sf
    data = np.zeros_like(time)
    # Random state of the machine :
    rng = np.random.RandomState(rnd_state)
    # Band / Delta parameters :
    sh = (ntrials, 1)
    if fpha.ndim == 0:
        apha = [fpha * (1 - dpha / 100), fpha * (1 + dpha / 100)]
        del_pha = apha[0] + (apha[1] - apha[0]) * rng.rand(*sh)
    elif fpha.ndim == 1:
        del_pha = rng.uniform(fpha[0], fpha[1], ntrials)
    if famp.ndim == 0:
        a_amp = [famp * (1 - damp / 100), famp * (1 + damp / 100)]
        del_amp = a_amp[0] + (a_amp[1] - a_amp[0]) * rng.rand(*sh)
    elif famp.ndim == 1:
        del_amp = rng.uniform(famp[0], famp[1], ntrials)

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
    data += noise * rng.rand(*data.shape)  # Add noise

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
    fpha, famp : tuple | (2, 30, 2, 1), (60, 200, 10, 5)
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
    fstart : float | 60.
        Starting frequency.

    fend : float | 160.
        Ending frequency.

    fwidth : float | 10.
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
