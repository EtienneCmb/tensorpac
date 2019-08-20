"""Utility functions."""
import logging

import numpy as np
from scipy.signal import periodogram

from tensorpac.methods.meth_pac import _kl_hr
from tensorpac.pac import _PacObj

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
        le = len(starting) - (num + 1)
        # Create the frequency vector for this starting frequency :
        fst = np.c_[np.full(le, k), starting[num + 1::]]
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
    x : array_like
        Array of data of shape (n_epochs, n_times)
    sf : float
        The sampling frequency.
    """

    def __init__(self, x, sf):
        """Init."""
        assert isinstance(x, np.ndarray) and (x.ndim == 2), (
            "x should be a 2d array of shape (n_epochs, n_times)")
        self._n_trials, self._n_times = x.shape
        logger.info(f"Compute PSD over {self._n_trials} trials and "
                    f"{self._n_times} time points")
        self._freqs, self._psd = periodogram(x, fs=sf, window=None,
                                             nfft=self._n_times,
                                             detrend='constant',
                                             return_onesided=True,
                                             scaling='density', axis=1)

    def plot(self, f_min=None, f_max=None, confidence=95, interp=None,
             log=False, grid=True):
        """Plot the PSD.

        Parameters
        ----------
        f_min, f_max : (int, float) | None
            Frequency bounds to use for plotting
        confidence : (int, float) | None
            Light gray confidence interval. If None, no interval will be
            displayed
        interp : int | None
            Line interpolation integer. For example, if interp is 10 the number
            of points is going to be multiply by 10
        log : bool | False
            Use a log scale representation
        grid : bool | True
            Add a grid to the plot

        Returns
        -------
        ax : Matplotlib axis
            The matplotlib axis that contains the figure
        """
        import matplotlib.pyplot as plt
        f_types = (int, float)
        # interpolation
        xvec, yvec = self._freqs, self._psd
        if isinstance(interp, int) and (interp > 1):
            # from scipy.interpolate import make_interp_spline, BSpline
            from scipy.interpolate import interp1d
            xnew = np.linspace(xvec[0], xvec[-1], len(xvec) * interp)
            f = interp1d(xvec, yvec, kind='quadratic', axis=1)
            yvec = f(xnew)
            xvec = xnew
        # (f_min, f_max)
        f_min = xvec[0] if not isinstance(f_min, f_types) else f_min
        f_max = xvec[-1] if not isinstance(f_max, f_types) else f_max
        # plot main psd
        plt.plot(xvec, yvec.mean(0), color='black',
                 label='mean PSD over trials')
        # plot confidence interval
        if isinstance(confidence, (int, float)) and (0 < confidence < 100):
            logger.info(f"    Add {confidence}th confidence interval")
            interval = (100. - confidence) / 2
            kw = dict(axis=0, interpolation='nearest')
            psd_min = np.percentile(yvec, interval, **kw)
            psd_max = np.percentile(yvec, 100. - interval, **kw)
            plt.fill_between(xvec, psd_max, psd_min, color='lightgray',
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
            plt.grid(color='grey', which='major', linestyle='-',
                     linewidth=1., alpha=0.5)
            plt.grid(color='lightgrey', which='minor', linestyle='--',
                     linewidth=0.5, alpha=0.5)

        return plt.gca()

    def show(self):
        """Display the PSD figure."""
        import matplotlib.pyplot as plt
        plt.show()

    @property
    def freqs(self):
        """Get the frequency vector."""
        return self._freqs

    @property
    def psd(self):
        """Get the psd value."""
        return self._psd


class BinAmplitude(_PacObj):
    """Bin the amplitude according to the phase.

    Parameters
    ----------
    x : array_like
        Array of data of shape (n_epochs, n_times)
    sf : float
        The sampling frequency
    f_pha : tuple, list | [2, 4]
        List of two floats describing the frequency bounds for extracting the
        phase
    f_amp : tuple, list | [60, 80]
        List of two floats describing the frequency bounds for extracting the
        amplitude
    n_bins : int | 18
        Number of bins to use to binarize the phase and the amplitude
    dcomplex : {'wavelet', 'hilbert'}
        Method for the complex definition. Use either 'hilbert' or
        'wavelet'.
    cycle : tuple | (3, 6)
        Control the number of cycles for filtering (only if dcomplex is
        'hilbert'). Should be a tuple of integers where the first one
        refers to the number of cycles for the phase and the second for the
        amplitude [#f5]_.
    width : int | 7
        Width of the Morlet's wavelet.
    edges : int | None
        Number of samples to discard to avoid edge effects due to filtering
    """

    def __init__(self, x, sf, f_pha=[2, 4], f_amp=[60, 80], n_bins=18,
                 dcomplex='hilbert', cycle=(3, 6), width=7, edges=None,
                 n_jobs=-1):
        """Init."""
        _PacObj.__init__(self, f_pha=f_pha, f_amp=f_amp, dcomplex=dcomplex,
                         cycle=cycle, width=width)
        # check
        x = np.atleast_2d(x)
        assert x.ndim <= 2, ("`x` input should be an array of shape "
                             "(n_epochs, n_times)")
        assert isinstance(sf, (int, float)), ("`sf` input should be a integer "
                                              "or a float")
        assert all([isinstance(k, (int, float)) for k in f_pha]), (
            "`f_pha` input should be a list of two integers / floats")
        assert all([isinstance(k, (int, float)) for k in f_amp]), (
            "`f_amp` input should be a list of two integers / floats")
        assert isinstance(n_bins, int), "`n_bins` should be an integer"
        # extract phase and amplitude
        kw = dict(keepfilt=False, edges=edges, n_jobs=n_jobs)
        pha = self.filter(sf, x, 'phase', **kw)
        amp = self.filter(sf, x, 'amplitude', **kw)
        # binarize amplitude according to phase
        self._amplitude = _kl_hr(pha, amp, n_bins).squeeze()
        self.n_bins = n_bins

    def plot(self, unit='rad', **kw):
        """Plot the amplitude.

        Parameters
        ----------
        unit : {'rad', 'deg'}
            The unit to use for the phase. Use either 'deg' for degree or 'rad'
            for radians
        kw : dict | {}
            Additional inputs are passed to the matplotlib.pyplot.bar function

        Returns
        -------
        ax : Matplotlib axis
            The matplotlib axis that contains the figure
        """
        import matplotlib.pyplot as plt
        assert unit in ['rad', 'deg']
        if unit == 'rad':
            self._phase = np.linspace(-np.pi, np.pi, self.n_bins)
            width = 2 * np.pi / self.n_bins
        elif unit == 'deg':
            self._phase = np.linspace(-180, 180, self.n_bins)
            width = 360 / self.n_bins
        plt.bar(self._phase, self._amplitude.mean(1), width=width, **kw)
        plt.xlabel(f"Frequency phase ({self.n_bins} bins)")
        plt.ylabel("Amplitude")
        plt.title("Binned amplitude")
        plt.autoscale(enable=True, axis='x', tight=True)

    def show(self):
        """Show the figure."""
        import matplotlib.pyplot as plt
        plt.show()

    @property
    def amplitude(self):
        """Get the amplitude value."""
        return self._amplitude

    @property
    def phase(self):
        """Get the phase value."""
        return self._phase


class PLV(_PacObj):
    """Compute the Phase Locking Value (PLV).

    The Phase Locking Value (PLV) can be used to measure the phase synchrony
    across trials [#f1]_

    Parameters
    ----------
    x : array_like
        Array of data of shape (n_epochs, n_times)
    sf : float
        The sampling frequency
    f_pha : tuple, list | [2, 4]
        List of two floats describing the frequency bounds for extracting the
        phase
    dcomplex : {'wavelet', 'hilbert'}
        Method for the complex definition. Use either 'hilbert' or
        'wavelet'.
    cycle : tuple | 3
        Control the number of cycles for filtering the phase (only if dcomplex
        is 'hilbert').
    width : int | 7
        Width of the Morlet's wavelet.
    edges : int | None
        Number of samples to discard to avoid edge effects due to filtering

    References
    ----------
    .. [#f1] `Lachaux et al, 1999 <https://onlinelibrary.wiley.com/doi/abs/10.
       1002/(SICI)1097-0193(1999)8:4%3C194::AID-HBM4%3E3.0.CO;2-C>`_
    """

    def __init__(self, x, sf, f_pha=[2, 4], dcomplex='hilbert', cycle=3,
                 width=7, edges=None, n_jobs=-1):
        """Init."""
        _PacObj.__init__(self, f_pha=f_pha, f_amp=[60, 80], dcomplex=dcomplex,
                         cycle=(cycle, 6), width=width)
        # check
        x = np.atleast_2d(x)
        assert x.ndim <= 2, ("`x` input should be an array of shape "
                             "(n_epochs, n_times)")
        self._n_trials = x.shape[0]
        # extract phase and amplitude
        kw = dict(keepfilt=False, edges=edges, n_jobs=n_jobs)
        pha = self.filter(sf, x, 'phase', **kw)
        # compute plv
        self._plv = np.abs(np.exp(1j * pha).mean(1)).squeeze()
        self._sf = sf

    def plot(self, time=None, **kw):
        """Plot the Phase Locking Value.

        Parameters
        ----------
        time : array_like | None
            Custom time vector to use
        kw : dict | {}
            Additional inputs are either pass to the matplotlib.pyplot.plot
            function if a single phase band is used, otherwise to the
            matplotlib.pyplot.pcolormesh function

        Returns
        -------
        ax : Matplotlib axis
            The matplotlib axis that contains the figure
        """
        import matplotlib.pyplot as plt
        n_pts = self._plv.shape[-1]
        if not isinstance(time, np.ndarray):
            time = np.arange(n_pts) / self._sf
        time = time[self._edges]
        assert len(time) == n_pts, ("The length of the time vector should be "
                                    "{n_pts}")
        if self._plv.ndim == 1:
            plt.plot(time, self._plv, **kw)
        elif self._plv.ndim == 2:
            vmin = kw.get('vmin', np.percentile(self._plv, 1))
            vmax = kw.get('vmax', np.percentile(self._plv, 99))
            plt.pcolormesh(time, self.xvec, self._plv, vmin=vmin, vmax=vmax,
                           **kw)
            plt.colorbar()
            plt.ylabel("Frequency for phase (Hz)")
        plt.xlabel('Time')
        plt.title(f"Phase Locking Value across {self._n_trials} trials")
        return plt.gca()

    def show(self):
        """Show the figure."""
        import matplotlib.pyplot as plt
        plt.show()

    @property
    def plv(self):
        """Get the plv value."""
        return self._plv
