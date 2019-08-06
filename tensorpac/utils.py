"""Utility functions."""
import logging

import numpy as np
from scipy.signal import periodogram

from tensorpac.spectral import morlet

logger = logging.getLogger('tensorpac')


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


class PSD(object):
    """Power Spectrum Density for electrophysiological brain data.

    Parameters
    ----------
    sf : float
        The sampling frequency.
    x : array_like
        Array of data of shape (n_epochs, n_times)
    """

    def __init__(self, x, sf):
        """Init."""
        assert isinstance(x, np.ndarray) and (x.ndim == 2), (
            "x should be a 2d array of shape (n_epochs, n_times)")
        self._n_trials, self._n_times = x.shape
        logger.info(f"Compute PSD over {self._n_trials} trials and "
                    f"{self._n_times} time points")
        self.freqs, self.psd = periodogram(x, fs=sf, window=None,
                                           nfft=self._n_times,
                                           detrend='constant',
                                           return_onesided=True,
                                           scaling='density', axis=1)

    def plot(self, f_min=None, f_max=None, confidence=95, log=False,
             grid=True):
        """Plot the PSD.

        Parameters
        ----------
        f_min, f_max : (int, float) | None
            Frequency bounds to use for plotting
        confidence : (int, float) | None
            Light gray confidence interval. If None, no interval will be
            displayed

        Returns
        -------
        ax : Matplotlib axis
            The matplotlib axis that contains the figure
        """
        import matplotlib.pyplot as plt
        f_types = (int, float)
        # psd mean and deviation
        psd_mean = self.psd.mean(0)

        # (f_min, f_max)
        f_min = self.freqs[0] if not isinstance(f_min, f_types) else f_min
        f_max = self.freqs[-1] if not isinstance(f_max, f_types) else f_max
        # plot main psd
        plt.plot(self.freqs, self.psd.mean(0), color='black',
                 label='mean PSD over trials')
        # plot confidence interval
        if isinstance(confidence, (int, float)) and (0 < confidence < 100):
            logger.info(f"    Add {confidence}th confidence interval")
            interval = (100. - confidence) / 2
            kw = dict(axis=0, interpolation='nearest')
            psd_min = np.percentile(self.psd, interval, **kw)
            psd_max = np.percentile(self.psd, 100. - interval, **kw)
            plt.fill_between(self.freqs, psd_max, psd_min, color='lightgray',
                             alpha=0.5,
                             label=f"{confidence}th confidence interval")
            plt.legend()
        plt.xlabel("Frequencies (Hz)"), plt.ylabel("Power (V**2/Hz)")  # noqa
        plt.title(f"PSD mean over {self._n_trials} trials")
        plt.xlim(f_min, f_max)
        if log:
            from matplotlib.ticker import ScalarFormatter
            plt.xscale('log', basex=10)
            plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        if grid:
            # plt.grid(True, which='both')
            plt.grid(color='grey', which='major', linestyle='-',
                     linewidth=1., alpha=0.5)
            plt.grid(color='lightgrey', which='minor', linestyle='--',
                     linewidth=0.5, alpha=0.5)

        return plt.gca()

    def show(self):
        """Display the PSD figure."""
        import matplotlib.pyplot as plt
        plt.show()
