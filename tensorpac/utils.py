"""Utility PAC functions.

- pac_signals_tort : generate artificially phase-amplitude coupled signals
- pac_vec : generate cross-frequency coupling vectors
"""
import numpy as np

from tensorpac.spectral import morlet


###############################################################################
###############################################################################
#                             SIGNALS
###############################################################################
###############################################################################


def pac_signals_wavelet(f_pha=10., f_amp=100., sf=1024., n_times=4000.,
                        n_epochs=10, noise=.1, pp=0., rnd_state=0):
    """Generate artificially phase-amplitude coupled signals using wavelets.

    This function is inspired by the code of the pactools toolbox developped by
    Tom Dupre la Tour.

    Parameters
    ----------
    f_pha : float | 10.
        Frequency for phase. Use either a float number for a centered frequency
        of a band (like [5, 7]) for a bandwidth.
    f_amp : float | 100.
        Frequency for amplitude. Use either a float number for a centered
        frequency of a band (like [60, 80]) for a bandwidth.
    sf : float | 1024.
        Sampling frequency.
    n_times : int | 4000
        Number of time points.
    n_epochs : int | 10
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
        Array of signals of shape (n_epochs, n_times).
    time : array_like
        Time vector of shape (n_times,).
    """
    n_times = int(n_times)
    sf = float(sf)
    f_pha, f_amp = np.asarray(f_pha).mean(), np.asarray(f_amp).mean()
    # time = np.mgrid[0:n_epochs, 0:n_times][1] / sf
    time = np.arange(n_times) / sf
    # Random state of the machine :
    rng = np.random.RandomState(rnd_state)
    # Get complex decomposition of random points in the phase frequency band :
    driver = morlet(rng.randn(n_epochs, n_times), sf, f_pha)
    driver /= np.max(driver, axis=1, keepdims=True)
    # Create amplitude signals :
    xh = np.sin(2 * np.pi * f_amp * time.reshape(1, -1))
    dpha = np.exp(-1j * pp)
    modulation = 1. / (1. + np.exp(- 6. * 1. * np.real(driver * dpha)))
    # Modulate the amplitude :
    xh = xh * modulation
    # Get the phase signal :
    xl = np.real(driver)
    # Build the pac signal :
    data = xh + xl + noise * rng.randn(*xh.shape)

    return data, time


def pac_signals_tort(f_pha=10., f_amp=100., sf=1024, n_times=4000, n_epochs=10,
                     chi=0., noise=1., dpha=0., damp=0., rnd_state=0):
    """Generate artificially phase-amplitude coupled signals.

    This function use the definition of Tort et al. 2010.

    Parameters
    ----------
    f_pha : float | 10.
        Frequency for phase. Use either a float number for a centered frequency
        of a band (like [5, 7]) for a bandwidth.
    f_amp : float | 100.
        Frequency for amplitude. Use either a float number for a centered
        frequency of a band (like [60, 80]) for a bandwidth.
    sf : int | 1024
        Sampling frequency
    n_epochs : int | 10
        Number of datasets
    n_times : int | 4000
        Number of points for each signal.
    chi : float | 0.
        Amount of coupling. If chi=0, signals of phase and amplitude
        are strongly coupled (0.<=chi<=1.).
    noise : float | 1.
        Amount of noise (0<=noise<=3).
    dpha : float | 0.
        Random incertitude on phase frequences (0<=dpha<=100). If f_pha is 2,
        and dpha is 50, the frequency for the phase signal will be between :
        [2-0.5*2, 2+0.5*2]=[1,3]
    damp : float | 0.
        Random incertitude on amplitude frequencies (0<=damp<=100). If f_amp is
        60, and damp is 10, the frequency for the amplitude signal will be
        between : [60-0.1*60, 60+0.1*60]=[54,66]
    rnd_state: int | 0
        Fix random of the machine (for reproducibility)

    Returns
    -------
    data : array_like
        Array of signals of shape (n_epochs, n_channels, n_times).
    time : array_like
        Time vector of shape (n_times,).
    """
    n_times, sf = int(n_times), float(sf)
    # Check the inputs variables :
    chi = 0 if not 0 <= chi <= 1 else chi
    noise = 0 if not 0 <= noise <= 3 else noise
    dpha = 0 if not 0 <= dpha <= 100 else dpha
    damp = 0 if not 0 <= damp <= 100 else damp
    f_pha, f_amp = np.asarray(f_pha), np.asarray(f_amp)
    # time = np.mgrid[0:n_epochs, 0:n_times][1] / sf
    time = np.arange(n_times) / sf
    # Random state of the machine :
    rng = np.random.RandomState(rnd_state)
    # Band / Delta parameters :
    sh = (n_epochs, 1)
    if f_pha.ndim == 0:
        apha = [f_pha * (1. - dpha / 100.), f_pha * (1. + dpha / 100.)]
        del_pha = apha[0] + (apha[1] - apha[0]) * rng.rand(*sh)
    elif f_pha.ndim == 1:
        del_pha = rng.uniform(f_pha[0], f_pha[1], (n_epochs, 1))
    if f_amp.ndim == 0:
        a_amp = [f_amp * (1. - damp / 100.), f_amp * (1. + damp / 100.)]
        del_amp = a_amp[0] + (a_amp[1] - a_amp[0]) * rng.rand(*sh)
    elif f_amp.ndim == 1:
        del_amp = rng.uniform(f_amp[0], f_amp[1], (n_epochs, 1))

    # Create phase and amplitude signals :
    xl = np.sin(2 * np.pi * del_pha * time.reshape(1, -1))
    xh = np.sin(2 * np.pi * del_amp * time.reshape(1, -1))

    # Create the coupling :
    ah = .5 * ((1. - chi) * xl + 1. + chi)
    al = 1.

    # Generate datasets :
    data = (ah * xh) + (al * xl)
    data += noise * rng.rand(*data.shape)  # Add noise

    return data, time

###############################################################################
###############################################################################
#                             FREQUENCY VECTOR
###############################################################################
###############################################################################


def pac_vec(f_pha=(2, 30, 2, 1), f_amp=(60, 200, 10, 5)):
    """Generate cross-frequency coupling vectors.

    Parameters
    ----------
    f_pha, f_amp : tuple | (2, 30, 2, 1), (60, 200, 10, 5)
        Frequency parameters for phase and amplitude. Each argument inside the
        tuple mean (starting fcy, ending fcy, bandwidth, step).

    Returns
    -------
    f_pha, f_amp : array_like
        Arrays containing the pairs of phase and amplitude frequencies. Each
        vector have a shape of (N, 2).
    """
    if isinstance(f_pha, str):
        if f_pha == 'lres':    # low resolution phase
            f_pha = (2, 30, 2, 2)
        elif f_pha == 'mres':  # middle resolution phase
            f_pha = (2, 30, 2, 1)
        elif f_pha == 'hres':  # high resolution phase
            f_pha = (2, 30, 1, .5)
    if isinstance(f_amp, str):
        if f_amp == 'lres':    # low resolution amplitude
            f_amp = (60, 160, 10, 10)
        elif f_amp == 'mres':  # middle resolution amplitude
            f_amp = (60, 160, 5, 4)
        elif f_amp == 'hres':  # high resolution amplitude
            f_amp = (60, 160, 4, 2)
    return _check_freq(f_pha), _check_freq(f_amp)


def _check_freq(f):
    """Check the frequency definition."""
    f = np.atleast_2d(np.asarray(f))
    #
    if len(f.reshape(-1)) == 1:
        raise ValueError("The length of f should at least be 2.")
    elif 2 in f.shape:  # f of shape (N, 2) or (2, N)
        if f.shape[1] is not 2:
            f = f.T
    elif np.squeeze(f).shape == (4,):  # (f_start, f_end, f_width, f_step)
        f = _pair_vectors(*tuple(np.squeeze(f)))
    else:  # Sequential
        f = f.reshape(-1)
        f.sort()
        f = np.c_[f[0:-1], f[1::]]
    return f


def _pair_vectors(f_start, f_end, f_width, f_step):
    # Generate two array for phase and amplitude :
    fdown = np.arange(f_start, f_end - f_width, f_step)
    fup = np.arange(f_start + f_width, f_end, f_step)
    return np.c_[fdown, fup]


def pac_trivec(f_start=60., f_end=160., f_width=10.):
    """Generate triangular vector.

    By contrast with the pac_vec function, this function generate frequency
    vector with an increasing frequency bandwidth.

    Parameters
    ----------
    f_start : float | 60.
        Starting frequency.
    f_end : float | 160.
        Ending frequency.
    f_width : float | 10.
        Frequency bandwidth increase between each band.

    Returns
    -------
    f : array_like
        The triangular vector.
    tridx : array_like
        The triangular index for the reconstruction.
    """
    starting = np.arange(f_start, f_end + f_width, f_width)
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
