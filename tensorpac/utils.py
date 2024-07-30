"""Utility functions."""

import logging

import numpy as np
from scipy.signal import periodogram

from tensorpac.io import set_log_level
from tensorpac.methods.meth_pac import _kl_hr
from tensorpac.pac import _PacObj, _PacVisual

logger = logging.getLogger("tensorpac")


def pac_vec(f_pha="mres", f_amp="mres"):
    """Generate cross-frequency coupling vectors.

    Parameters
    ----------
    Frequency vector for the phase and amplitude. Here you can use
    several forms to define those vectors :

        * Basic list/tuple (ex: [2, 4] or [8, 12]...)
        * List of frequency bands (ex: [[2, 4], [5, 7]]...)
        * Dynamic definition : (start, stop, width, step)
        * Range definition (ex : np.arange(3) => [[0, 1], [1, 2]])
        * Using a string. `f_pha` and `f_amp` can be 'lres', 'mres', 'hres'
          respectively for low, middle and high resolution vectors. In that
          case, it uses the definition proposed by Bahramisharif et al. 2013
          :cite:`bahramisharif2013propagating` i.e
          f_pha = [f - f / 4, f + f / 4] and f_amp = [f - f / 8, f + f / 8]

    Returns
    -------
    f_pha, f_amp : array_like
        Arrays containing the pairs of phase and amplitude frequencies. Each
        vector have a shape of (N, 2).
    """
    nb_fcy = dict(lres=10, mres=30, hres=50, demon=70, hulk=100)
    if isinstance(f_pha, str):
        # get where phase frequencies start / finish / number
        f_pha_start, f_pha_end = 2, 20
        f_pha_nb = nb_fcy[f_pha]
        # f_pha = [f - f / 4, f + f / 4]
        f_pha_mid = np.linspace(f_pha_start, f_pha_end, f_pha_nb)
        f_pha = np.c_[f_pha_mid - f_pha_mid / 4.0, f_pha_mid + f_pha_mid / 4.0]
    if isinstance(f_amp, str):
        # get where amplitude frequencies start / finish / number
        f_amp_start, f_amp_end = 60, 160
        f_amp_nb = nb_fcy[f_amp]
        # f_amp = [f - f / 8, f + f / 8]
        f_amp_mid = np.linspace(f_amp_start, f_amp_end, f_amp_nb)
        f_amp = np.c_[f_amp_mid - f_amp_mid / 8.0, f_amp_mid + f_amp_mid / 8.0]

    return _check_freq(f_pha), _check_freq(f_amp)


def _check_freq(f):
    """Check the frequency definition."""
    f = np.atleast_2d(np.asarray(f))
    #
    if len(f.reshape(-1)) == 1:
        raise ValueError("The length of f should at least be 2.")
    elif 2 in f.shape:  # f of shape (N, 2) or (2, N)
        if f.shape[1] != 2:
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


def pac_trivec(f_start=60.0, f_end=160.0, f_width=10.0):
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
        fst = np.c_[np.full(le, k), starting[num + 1 : :]]
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
        assert isinstance(x, np.ndarray) and (
            x.ndim == 2
        ), "x should be a 2d array of shape (n_epochs, n_times)"
        self._n_trials, self._n_times = x.shape
        logger.info(
            f"Compute PSD over {self._n_trials} trials and "
            f"{self._n_times} time points"
        )
        self._freqs, self._psd = periodogram(
            x,
            fs=sf,
            window=None,
            nfft=self._n_times,
            detrend="constant",
            return_onesided=True,
            scaling="density",
            axis=1,
        )

    def plot(
        self,
        f_min=None,
        f_max=None,
        confidence=95,
        interp=None,
        log=False,
        grid=True,
        fz_title=18,
        fz_labels=15,
    ):
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
        fz_title : int | 18
            Font size for the title
        fz_labels : int | 15
            Font size the x/y labels

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
            f = interp1d(xvec, yvec, kind="quadratic", axis=1)
            yvec = f(xnew)
            xvec = xnew
        # (f_min, f_max)
        f_min = xvec[0] if not isinstance(f_min, f_types) else f_min
        f_max = xvec[-1] if not isinstance(f_max, f_types) else f_max
        # plot main psd
        plt.plot(xvec, yvec.mean(0), color="black", label="mean PSD over trials")
        # plot confidence interval
        if isinstance(confidence, (int, float)) and (0 < confidence < 100):
            logger.info(f"    Add {confidence}th confidence interval")
            interval = (100.0 - confidence) / 2
            kw = dict(axis=0, interpolation="nearest")
            psd_min = np.percentile(yvec, interval, **kw)
            psd_max = np.percentile(yvec, 100.0 - interval, **kw)
            plt.fill_between(
                xvec,
                psd_max,
                psd_min,
                color="lightgray",
                alpha=0.5,
                label=f"{confidence}th confidence interval",
            )
            plt.legend(fontsize=fz_labels)
        plt.xlabel("Frequencies (Hz)", fontsize=fz_labels)
        plt.ylabel("Power (V**2/Hz)", fontsize=fz_labels)
        plt.title(f"PSD mean over {self._n_trials} trials", fontsize=fz_title)
        plt.xlim(f_min, f_max)
        if log:
            from matplotlib.ticker import ScalarFormatter

            plt.xscale("log", base=10)
            plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        if grid:
            plt.grid(
                color="grey", which="major", linestyle="-", linewidth=1.0, alpha=0.5
            )
            plt.grid(
                color="lightgrey",
                which="minor",
                linestyle="--",
                linewidth=0.5,
                alpha=0.5,
            )

        return plt.gca()

    def plot_st_psd(
        self,
        f_min=None,
        f_max=None,
        log=False,
        grid=True,
        fz_title=18,
        fz_labels=15,
        fz_cblabel=15,
        **kw,
    ):
        """Single-trial PSD plot.

        Parameters
        ----------
        f_min, f_max : (int, float) | None
            Frequency bounds to use for plotting
        log : bool | False
            Use a log scale representation
        grid : bool | True
            Add a grid to the plot
        fz_title : int | 18
            Font size for the title
        fz_labels : int | 15
            Font size the x/y labels
        fz_cblabel : int | 15
            Font size the colorbar label labels

        Returns
        -------
        ax : Matplotlib axis
            The matplotlib axis that contains the figure
        """
        import matplotlib.pyplot as plt

        # manage input variables
        kw["fz_labels"] = kw.get("fz_labels", fz_labels)
        kw["fz_title"] = kw.get("fz_title", fz_title)
        kw["fz_cblabel"] = kw.get("fz_cblabel", fz_title)
        kw["xlabel"] = kw.get("xlabel", "Frequencies (Hz)")
        kw["ylabel"] = kw.get("ylabel", "Trials")
        kw["title"] = kw.get("title", "Single-trial PSD")
        kw["cblabel"] = kw.get("cblabel", "Power (V**2/Hz)")
        # (f_min, f_max)
        xvec, psd = self._freqs, self._psd
        f_types = (int, float)
        f_min = xvec[0] if not isinstance(f_min, f_types) else f_min
        f_max = xvec[-1] if not isinstance(f_max, f_types) else f_max
        # locate (f_min, f_max) indices
        f_min_idx = np.abs(xvec - f_min).argmin()
        f_max_idx = np.abs(xvec - f_max).argmin()
        sl_freq = slice(f_min_idx, f_max_idx)
        xvec = xvec[sl_freq]
        psd = psd[:, sl_freq]
        # make the 2D plot
        _viz = _PacVisual()
        trials = np.arange(self._n_trials)
        _viz.pacplot(psd, xvec, trials, **kw)
        if log:
            from matplotlib.ticker import ScalarFormatter

            plt.xscale("log", base=10)
            plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        if grid:
            plt.grid(
                color="grey", which="major", linestyle="-", linewidth=1.0, alpha=0.5
            )
            plt.grid(
                color="lightgrey",
                which="minor",
                linestyle="--",
                linewidth=0.5,
                alpha=0.5,
            )

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
        amplitude :cite:`bahramisharif2013propagating`.
    width : int | 7
        Width of the Morlet's wavelet.
    edges : int | None
        Number of samples to discard to avoid edge effects due to filtering
    """

    def __init__(
        self,
        x,
        sf,
        f_pha=[2, 4],
        f_amp=[60, 80],
        n_bins=18,
        dcomplex="hilbert",
        cycle=(3, 6),
        width=7,
        edges=None,
        n_jobs=-1,
    ):
        """Init."""
        _PacObj.__init__(
            self, f_pha=f_pha, f_amp=f_amp, dcomplex=dcomplex, cycle=cycle, width=width
        )
        # check
        x = np.atleast_2d(x)
        assert x.ndim <= 2, (
            "`x` input should be an array of shape " "(n_epochs, n_times)"
        )
        assert isinstance(sf, (int, float)), (
            "`sf` input should be a integer " "or a float"
        )
        assert all(
            [isinstance(k, (int, float)) for k in f_pha]
        ), "`f_pha` input should be a list of two integers / floats"
        assert all(
            [isinstance(k, (int, float)) for k in f_amp]
        ), "`f_amp` input should be a list of two integers / floats"
        assert isinstance(n_bins, int), "`n_bins` should be an integer"
        logger.info(f"Binning {f_amp}Hz amplitude according to {f_pha}Hz " "phase")
        # extract phase and amplitude
        kw = dict(keepfilt=False, edges=edges, n_jobs=n_jobs)
        pha = self.filter(sf, x, "phase", **kw)
        amp = self.filter(sf, x, "amplitude", **kw)
        # binarize amplitude according to phase
        self._amplitude = _kl_hr(pha, amp, n_bins, mean_bins=False).squeeze()
        self.n_bins = n_bins

    def plot(self, unit="rad", normalize=False, **kw):
        """Plot the amplitude.

        Parameters
        ----------
        unit : {'rad', 'deg'}
            The unit to use for the phase. Use either 'deg' for degree or 'rad'
            for radians
        normalize : bool | None
            Normalize the histogram by the maximum
        kw : dict | {}
            Additional inputs are passed to the matplotlib.pyplot.bar function

        Returns
        -------
        ax : Matplotlib axis
            The matplotlib axis that contains the figure
        """
        import matplotlib.pyplot as plt

        assert unit in ["rad", "deg"]
        if unit == "rad":
            self._phase = np.linspace(-np.pi, np.pi, self.n_bins)
            width = 2 * np.pi / self.n_bins
        elif unit == "deg":
            self._phase = np.linspace(-180, 180, self.n_bins)
            width = 360 / self.n_bins
        amp_mean = self._amplitude.mean(1)
        if normalize:
            amp_mean /= amp_mean.max()
        plt.bar(self._phase, amp_mean, width=width, **kw)
        plt.xlabel(f"Frequency phase ({self.n_bins} bins)", fontsize=18)
        plt.ylabel("Amplitude", fontsize=18)
        plt.title("Binned amplitude")
        plt.autoscale(enable=True, axis="x", tight=True)

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


class ITC(_PacObj, _PacVisual):
    """Compute the Inter-Trials Coherence (ITC).

    The Inter-Trials Coherence (ITC) is a measure of phase consistency over
    trials for a single recording site (electrode / sensor etc.).

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
    """

    def __init__(
        self,
        x,
        sf,
        f_pha=[2, 4],
        dcomplex="hilbert",
        cycle=3,
        width=7,
        edges=None,
        n_jobs=-1,
        verbose=None,
    ):
        """Init."""
        set_log_level(verbose)
        _PacObj.__init__(
            self,
            f_pha=f_pha,
            f_amp=[60, 80],
            dcomplex=dcomplex,
            cycle=(cycle, 6),
            width=width,
        )
        _PacVisual.__init__(self)
        # check
        x = np.atleast_2d(x)
        assert x.ndim <= 2, (
            "`x` input should be an array of shape " "(n_epochs, n_times)"
        )
        self._n_trials = x.shape[0]
        logger.info("Inter-Trials Coherence (ITC)")
        logger.info(f"    extracting {len(self.xvec)} phases")
        # extract phase and amplitude
        kw = dict(keepfilt=False, edges=edges, n_jobs=n_jobs)
        pha = self.filter(sf, x, "phase", **kw)
        # compute itc
        self._itc = np.abs(np.exp(1j * pha).mean(1)).squeeze()
        self._sf = sf

    def plot(self, times=None, **kw):
        """Plot the Inter-Trials Coherence.

        Parameters
        ----------
        times : array_like | None
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

        n_pts = self._itc.shape[-1]
        if not isinstance(times, np.ndarray):
            times = np.arange(n_pts) / self._sf
        times = times[self._edges]
        assert len(times) == n_pts, "The length of the time vector should be " "{n_pts}"
        xlab = "Time"
        title = f"Inter-Trials Coherence ({self._n_trials} trials)"
        if self._itc.ndim == 1:
            plt.plot(times, self._itc, **kw)
        elif self._itc.ndim == 2:
            vmin = kw.get("vmin", np.percentile(self._itc, 1))
            vmax = kw.get("vmax", np.percentile(self._itc, 99))
            self.pacplot(
                self._itc,
                times,
                self.xvec,
                vmin=vmin,
                vmax=vmax,
                ylabel="Frequency for phase (Hz)",
                xlabel=xlab,
                title=title,
                **kw,
            )
        return plt.gca()

    def show(self):
        """Show the figure."""
        import matplotlib.pyplot as plt

        plt.show()

    @property
    def itc(self):
        """Get the itc value."""
        return self._itc


class PeakLockedTF(_PacObj, _PacVisual):
    """Peak-Locked Time-frequency representation.

    This class can be used in order to re-align time-frequency representations
    around a time-point (cue) according to the closest phase peak. This type
    of visualization can bring out a cyclic behavior of the amplitude at a
    given phase, potentially indicating the presence of a phase-amplitude
    coupling. Here's the detailed pipeline :

        * Filter around a single phase frequency bands and across multiple
          amplitude frequencies
        * Use a `cue` which define the time-point to use for the realignment
        * Detect in the filtered phase the closest peak to the cue. This step
          is repeated to each trial in order to get a list of length (n_epochs)
          that contains the number of sample (shift) so that if the phase is
          moved, the peak fall onto the cue. A positive shift indicates that
          the phase is moved forward while a negative shift is for a backward
          move
        * Apply, to each trial, this shift to the amplitude
        * Plot the mean re-aligned amplitudes

    Parameters
    ----------
    x : array_like
        Array of data of shape (n_epochs, n_times)
    sf : float
        The sampling frequency
    cue : int, float
        Time-point to use in order to detect the closest phase peak. This
        parameter works in conjunction with the `times` input below. Use
        either :

            * An integer and `times` is None to indicate that you want to
              realign according to a time-point in sample
            * A integer or a float with `times` the time vector if you want
              that Tensorpac automatically infer the sample number around which
              to align
    times : array_like | None
        Time vector
    f_pha : tuple, list | [2, 4]
        List of two floats describing the frequency bounds for extracting the
        phase
    f_amp : tuple, list | [60, 80]
        Frequency vector for the amplitude. Here you can use several forms to
        define those vectors :

            * Dynamic definition : (start, stop, width, step)
            * Using a string : `f_amp` can be 'lres', 'mres', 'hres'
              respectively for low, middle and high resolution vectors
    cycle : tuple | (3, 6)
        Control the number of cycles for filtering. Should be a tuple of
        integers where the first one refers to the number of cycles for the
        phase and the second for the amplitude
        :cite:`bahramisharif2013propagating`.
    """

    def __init__(
        self,
        x,
        sf,
        cue,
        times=None,
        f_pha=[5, 7],
        f_amp="hres",
        cycle=(3, 6),
        n_jobs=-1,
        verbose=None,
    ):
        """Init."""
        set_log_level(verbose)
        # initialize to retrieve filtering methods
        _PacObj.__init__(
            self, f_pha=f_pha, f_amp=f_amp, dcomplex="hilbert", cycle=cycle
        )
        _PacVisual.__init__(self)
        logger.info("PeakLockedTF object defined")
        # inputs checking
        x = np.atleast_2d(x)
        assert isinstance(x, np.ndarray) and (x.ndim == 2)
        assert isinstance(sf, (int, float))
        assert isinstance(cue, (int, float))
        assert isinstance(f_pha, (list, tuple)) and (len(f_pha) == 2)
        n_epochs, n_times = x.shape

        # manage cur conversion
        if times is None:
            cue = int(cue)
            times = np.arange(n_times)
            logger.info(f"    align on sample cue={cue}")
        else:
            assert isinstance(times, np.ndarray) and (len(times) == n_times)
            cue_time = cue
            cue = np.abs(times - cue).argmin() - 1
            logger.info(f"    align on time-point={cue_time} (sample={cue})")
        self.cue, self._times = cue, times

        # extract phase and amplitudes
        logger.info(f"    extract phase and amplitudes " f"(n_amps={len(self.yvec)})")
        kw = dict(keepfilt=False, n_jobs=n_jobs)
        pha = self.filter(sf, x, "phase", n_jobs=n_jobs, keepfilt=True)
        amp = self.filter(sf, x, "amplitude", n_jobs=n_jobs)
        self._pha, self._amp = pha, amp**2

        # peak detection
        logger.info(f"    running peak detection around sample={cue}")
        self.shifts = self._peak_detection(self._pha.squeeze(), cue)

        # realign phases and amplitudes
        logger.info(f"    realign the {n_epochs} phases and amplitudes")
        self.amp_a = self._shift_signals(self._amp, self.shifts, fill_with=0.0)
        self.pha_a = self._shift_signals(self._pha, self.shifts, fill_with=0.0)

    @staticmethod
    def _peak_detection(pha, cue):
        """Single trial closest to a cue peak detection.

        Parameters
        ----------
        pha : array_like
            Array of single trial phases of shape (n_trials, n_times)
        cue : int
            Cue to use as a reference (in sample unit)

        Returns
        -------
        peaks : array_like
            Array of length (n_trials,) describing each delay to apply
            to each trial in order to realign the phases. In detail :

                * Positive delays means that zeros should be prepend
                * Negative delays means that zeros should be append
        """
        n_trials, n_times = pha.shape
        peaks = []
        for tr in range(n_trials):
            # select the single trial phase
            st_pha = pha[tr, :]
            # detect all peaks across time points
            st_peaks = []
            for t in range(n_times - 1):
                if (st_pha[t - 1] < st_pha[t]) and (st_pha[t] > st_pha[t + 1]):
                    st_peaks += [t]
            # detect the minimum peak
            min_peak = st_peaks[np.abs(np.array(st_peaks) - cue).argmin()]
            peaks += [cue - min_peak]

        return np.array(peaks)

    @staticmethod
    def _shift_signals(sig, n_shifts, fill_with=0):
        """Shift an array of signals according to an array of delays.

        Parameters
        ----------
        sig : array_like
            Array of signals of shape (n_freq, n_trials, n_times)
        n_shifts : array_like
            Array of delays to apply to each trial of shape (n_trials,)
        fill_with : int
            Value to prepend / append to each shifted time-series

        Returns
        -------
        sig_shifted : array_like
            Array of shifted signals with the same shape as the input
        """
        # prepare the needed variables
        n_freqs, n_trials, n_pts = sig.shape
        sig_shifted = np.zeros_like(sig)
        # shift each trial
        for tr in range(n_trials):
            # select the data of a specific trial
            st_shift = n_shifts[tr]
            st_sig = sig[:, tr, :]
            fill = np.full((n_freqs, abs(st_shift)), fill_with, dtype=st_sig.dtype)
            # shift this specific trial
            if st_shift > 0:  # move forward = prepend zeros
                sig_shifted[:, tr, :] = np.c_[fill, st_sig][:, 0:-st_shift]
            elif st_shift < 0:  # move backward = append zeros
                sig_shifted[:, tr, :] = np.c_[st_sig, fill][:, abs(st_shift) :]

        return sig_shifted

    def plot(self, zscore=False, baseline=None, edges=0, **kwargs):
        """Integrated Peak-Locked TF plotting function.

        Parameters
        ----------
        zscore : bool | False
            Normalize the power by using a z-score normalization. This can be
            useful in order to compensate the 1 / f effect in the power
            spectrum. If True, the mean and deviation are computed at the
            single trial level and across all time points
        baseline : tuple | None
            Baseline period to use in order to apply the z-score correction.
            Should be in samples.
        edges : int | 0
            Number of pixels to discard to compensate filtering edge effect
            (`power[edges:-edges]`).
        kwargs : dict | {}
            Additional arguments are sent to the
            :class:`tensorpac.utils.PeakLockedTF.pacplot` method
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        # manage additional arguments
        kwargs["colorbar"] = False
        kwargs["ylabel"] = "Frequency for amplitude (hz)"
        kwargs["xlabel"] = ""
        kwargs["fz_labels"] = kwargs.get("fz_labels", 14)
        kwargs["fz_cblabel"] = kwargs.get("fz_cblabel", 14)
        kwargs["fz_title"] = kwargs.get("fz_title", 16)
        sl_times = slice(edges, len(self._times) - edges)
        times = self._times[sl_times]
        pha_n = self.pha_a[..., sl_times].squeeze()
        # z-score normalization
        if zscore:
            if baseline is None:
                bsl_idx = sl_times
            else:
                assert len(baseline) == 2
                bsl_idx = slice(baseline[0], baseline[1])
            _mean = self.amp_a[..., bsl_idx].mean(2, keepdims=True)
            _std = self.amp_a[..., bsl_idx].std(2, keepdims=True)
            _std[_std == 0.0] = 1.0  # correction from NaN
            amp_n = (self.amp_a[..., sl_times] - _mean) / _std
        else:
            amp_n = self.amp_a[..., sl_times]

        # grid definition
        gs = GridSpec(8, 8)
        # image plot
        plt.subplot(gs[slice(0, 6), 0:-1])
        self.pacplot(amp_n.mean(1), times, self.yvec, **kwargs)
        plt.axvline(times[self.cue], color="w", lw=2)
        plt.tick_params(bottom=False, labelbottom=False)
        ax_1 = plt.gca()
        # external colorbar
        plt.subplot(gs[slice(1, 5), -1])
        cb = plt.colorbar(self._plt_im, pad=0.01, cax=plt.gca())
        cb.set_label("Power (V**2/Hz)", fontsize=kwargs["fz_cblabel"])
        cb.outline.set_visible(False)
        # phase plot
        plt.subplot(gs[slice(6, 8), 0:-1])
        plt.plot(times, pha_n.T, color="lightgray", alpha=0.2, lw=1.0)
        plt.plot(
            times, pha_n.mean(0), label="single trial phases", alpha=0.2, lw=1.0
        )  # legend tweaking
        plt.plot(times, pha_n.mean(0), label="mean phases", color="#1f77b4")
        plt.axvline(times[self.cue], color="k", lw=2)
        plt.autoscale(axis="both", tight=True, enable=True)
        plt.xlabel("Times", fontsize=kwargs["fz_labels"])
        plt.ylabel("V / Hz", fontsize=kwargs["fz_labels"])
        # bottom legend
        plt.legend(loc="center", bbox_to_anchor=(0.5, -0.5), fontsize="x-large", ncol=2)
        ax_2 = plt.gca()

        return [ax_1, ax_2]
