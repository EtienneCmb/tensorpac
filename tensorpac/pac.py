"""Main PAC class."""
import numpy as np
import logging

from tensorpac.spectral import spectral, hilbertm
from tensorpac.methods import (get_pac_fcn, pacstr, compute_surrogates,
                               erpac, ergcpac, _ergcpac_perm, preferred_phase,
                               normalize)
from tensorpac.gcmi import copnorm
from tensorpac.visu import _PacVisual, _PacPlt, _PolarPlt
from tensorpac.io import set_log_level
from tensorpac.config import CONFIG

logger = logging.getLogger('tensorpac')


class _PacObj(object):
    """Main class for relative PAC objects."""

    def __init__(self, f_pha=[2, 4], f_amp=[60, 200], dcomplex='hilbert',
                 cycle=(3, 6), width=7):
        # Frequency checking :
        from tensorpac.utils import pac_vec
        self._f_pha, self._f_amp = pac_vec(f_pha, f_amp)
        self._xvec, self._yvec = self.f_pha.mean(1), self.f_amp.mean(1)
        # Check spectral properties :
        self._speccheck(dcomplex, cycle, width)

    def __str__(self):
        """String representation."""
        return self.method

    def filter(self, sf, x, ftype='phase', keepfilt=False, edges=None,
               n_jobs=-1):
        """Filt the data in the specified frequency bands.

        Parameters
        ----------
        sf : float
            The sampling frequency.
        x : array_like
            Array of data of shape (n_epochs, n_times)
        ftype : {'phase', 'amplitude'}
            Specify if you want to extract phase ('phase') or the amplitude
            ('amplitude').
        n_jobs : int | -1
            Number of jobs to compute PAC in parallel. For very large data,
            set this parameter to 1 in order to prevent large memory usage.
        keepfilt : bool | False
            Specify if you only want the filtered data (True). This parameter
            is only available with dcomplex='hilbert' and not wavelet.
        edges : int | None
            Number of samples to discard to avoid edge effects due to filtering

        Returns
        -------
        xfilt : array_like
            The filtered data of shape (n_freqs, n_epochs, n_times)
        """
        # ---------------------------------------------------------------------
        # check inputs
        assert isinstance(sf, (int, float)), ("The sampling frequency must be "
                                              "a float number.")
        # Compatibility between keepfilt and wavelet :
        if (keepfilt is True) and (self._dcomplex is 'wavelet'):
            raise ValueError("Using wavelet for the complex decomposition do "
                             "not allow to get filtered data only. Set the "
                             "keepfilt parameter to False or set dcomplex to "
                             "'hilbert'.")
        assert ftype in ['phase', 'amplitude'], ("ftype must either be 'phase'"
                                                 " or 'amplitude.'")
        mne_epochs_type = CONFIG['MNE_EPOCHS_TYPE']
        if not isinstance(x, np.ndarray) and type(x) in mne_epochs_type:
            x = x.get_data()
            sf = x.info['sfreq']
        if x.ndim == 1:
            x = x[np.newaxis, :]
        assert x.ndim == 2, ("x should be a 2d array like (n_epochs, n_times)")
        # check edges
        if not isinstance(edges, int):
            edges = slice(None)
        else:
            logger.debug(f"    Edges {edges} time samples ignored")
            edges = slice(edges, -edges)
        self._edges = edges

        # ---------------------------------------------------------------------
        # Switch between phase or amplitude :
        if ftype is 'phase':
            tosend = 'pha' if not keepfilt else None
            xfilt = spectral(x, sf, self.f_pha, tosend, self._dcomplex,
                             self._cycle[0], self._width, n_jobs)
        elif ftype is 'amplitude':
            tosend = 'amp' if not keepfilt else None
            xfilt = spectral(x, sf, self.f_amp, tosend, self._dcomplex,
                             self._cycle[1], self._width, n_jobs)
        return xfilt[..., edges]

    def _speccheck(self, dcomplex=None, cycle=None, width=None):
        """Check spectral parameters."""
        # Check cycle :
        if cycle is not None:
            cycle = np.asarray(cycle)
            if (len(cycle) is not 2) or not cycle.dtype == int:
                raise ValueError("Cycle must be a tuple of two integers.")
            else:
                self._cycle = cycle
        # Check complex decomposition :
        if dcomplex is not None:
            if dcomplex not in ['hilbert', 'wavelet']:
                raise ValueError("dcomplex must either be 'hilbert' or "
                                 "'wavelet'.")
            else:
                self._dcomplex = dcomplex
        # Convert Morlet's width :
        if width is not None:
            self._width = int(width)

    @staticmethod
    def _phampcheck(pha, amp):
        """Check phase and amplitude values."""
        assert pha.ndim == 3, ("`pha` should have a shape of (n_pha, n_epochs,"
                               " n_times)")
        assert amp.ndim == 3, ("`amp` should have a shape of (n_pha, n_epochs,"
                               " n_times)")
        assert pha.shape[1:] == amp.shape[1:], ("`pha` and `amp` must have the"
                                                " same number of trials, "
                                                "channels and time points")
        assert np.ptp(pha) <= 2 * np.pi, ("Your phase is probably in degrees "
                                          "and should be converted in radians "
                                          "using either np.degrees or "
                                          "np.deg2rad.")
        # Force the phase to be in [-pi, pi] :
        pha = (pha + np.pi) % (2. * np.pi) - np.pi
        return pha, amp

    def _infer_pvalues(self, effect, perm, p=.05, mcp='maxstat'):
        """Global function for statistical inferences.

        In order to work this method requires :

            * effect = array like of shape (n_dims...)
            * perm = array like of shape (n_perm, n_dims...)
        """
        assert all([isinstance(k, np.ndarray) for k in (effect, perm)])
        n_perm = perm.shape[0]
        # compute the minimum number of required permutations
        n_perm_req = int(10. / p)
        if n_perm < n_perm_req:
            logger.warning(f"For inferences at p<{p}, it is recommended to per"
                           f"form at least n_perm={n_perm_req} permutations")

        # ---------------------------------------------------------------------
        logger.info(f"    infer p-values at (p={p}, mcp={mcp})")
        # computes the pvalues
        if mcp is 'maxstat':
            max_p = perm.reshape(n_perm, -1).max(1)[np.newaxis, ...]
            nb_over = (effect[..., np.newaxis] <= max_p).sum(-1)
            pvalues = nb_over / n_perm
            # non-signi. p-values are set to 1. and min(pvalues) = 1 / n_perm
            pvalues[pvalues >= p] = 1.
            pvalues = np.maximum(1. / n_perm, pvalues)
        elif mcp in ['fdr', 'bonferroni']:
            from mne.stats import fdr_correction, bonferroni_correction
            fcn = fdr_correction if mcp is 'fdr' else bonferroni_correction
            # compute the p-values
            pvalues = (effect[np.newaxis, ...] <= perm).sum(0) / n_perm
            pvalues = np.maximum(1. / n_perm, pvalues)
            # apply correction
            is_signi, pvalues = fcn(pvalues, alpha=p)
            pvalues[~is_signi] = 1.

        return pvalues

    @property
    def f_pha(self):
        """Vector of phases of shape (n_pha, 2)."""
        return self._f_pha

    @property
    def f_amp(self):
        """Vector of amplitudes of shape (n_amp, 2)."""
        return self._f_amp

    @property
    def xvec(self):
        """Vector of phases of shape (n_pha,) use for plotting."""
        return self._xvec

    @property
    def yvec(self):
        """Vector of amplitudes of shape (n_amp,) use for plotting."""
        return self._yvec

    # ----------- DCOMPLEX -----------
    @property
    def dcomplex(self):
        """Get the dcomplex value."""
        return self._dcomplex

    @dcomplex.setter
    def dcomplex(self, value):
        """Set dcomplex value."""
        self._speccheck(dcomplex=value)

    # ----------- CYCLE -----------
    @property
    def cycle(self):
        """Get the cycle value."""
        return self._cycle

    @cycle.setter
    def cycle(self, value):
        """Set cycle value."""
        self._speccheck(cycle=value)

    # ----------- WIDTH -----------
    @property
    def width(self):
        """Get the width value."""
        return self._width

    @width.setter
    def width(self, value):
        """Set width value."""
        self._width = value


class Pac(_PacObj, _PacPlt):
    """Compute Phase-Amplitude Coupling (PAC).

    Computing PAC is assessed in three steps : compute the real PAC, compute
    surrogates and finally, because PAC is very sensible to the noise, correct
    the real PAC by the surrogates. This implementation is modular i.e. it lets
    you choose among a large range of possible combinations.

    Parameters
    ----------
    idpac : tuple/list | (1, 1, 3)
        Choose the combination of methods to use in order to extract PAC.
        This tuple must be composed of three integers where each one them
        refer

        * First digit : refer to the pac method

            - 1 : Mean Vector Length (MVL) :cite:`canolty2006high`
              (see :func:`tensorpac.methods.mean_vector_length`)
            - 2 : Modulation Index (MI) :cite:`tort2010measuring`
              (see :func:`tensorpac.methods.modulation_index`)
            - 3 : Heights Ratio (HR) :cite:`lakatos2005oscillatory`
              (see :func:`tensorpac.methods.heights_ratio`)
            - 4 : ndPAC :cite:`ozkurt2012statistically`
              (see :func:`tensorpac.methods.norm_direct_pac`)
            - 5 : Phase-Locking Value (PLV)
              :cite:`penny2008testing,lachaux1999measuring`
              (see :func:`tensorpac.methods.phase_locking_value`)
            - 6 : Gaussian Copula PAC (GCPAC) :cite:`ince2017statistical`
              (see :func:`tensorpac.methods.gauss_cop_pac`)

        * Second digit : refer to the method for computing surrogates

            - 0 : No surrogates
            - 1 : Swap phase / amplitude across trials
              :cite:`tort2010measuring`
              (see :func:`tensorpac.methods.swap_pha_amp`)
            - 2 : Swap amplitude time blocks
              :cite:`bahramisharif2013propagating`
              (see :func:`tensorpac.methods.swap_blocks`)
            - 3 : Time lag :cite:`canolty2006high`
              (see :func:`tensorpac.methods.time_lag`)

        * Third digit : refer to the normalization method for correction

            - 0 : No normalization
            - 1 : Substract the mean of surrogates
            - 2 : Divide by the mean of surrogates
            - 3 : Substract then divide by the mean of surrogates
            - 4 : Z-score

    f_pha, f_amp : list/tuple/array | def: [2, 4] and [60, 200]
        Frequency vector for the phase and amplitude. Here you can use
        several forms to define those vectors :

            * Basic list/tuple (ex: [2, 4] or [8, 12]...)
            * List of frequency bands (ex: [[2, 4], [5, 7]]...)
            * Dynamic definition : (start, stop, width, step)
            * Range definition (ex : np.arange(3) => [[0, 1], [1, 2]])
            * Using a string. `f_pha` and `f_amp` can be 'lres', 'mres', 'hres'
              respectively for low, middle and high resolution vectors. In that
              case, it uses the definition proposed by Bahramisharif et al.
              2013 :cite:`bahramisharif2013propagating` i.e
              f_pha = [f - f / 4, f + f / 4] and f_amp = [f - f / 8, f + f / 8]

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
    n_bins : int | 18
        Number of bins for the KLD and HR PAC method
        :cite:`tort2010measuring,lakatos2005oscillatory`
    """

    def __init__(self, idpac=(1, 2, 3), f_pha=[2, 4], f_amp=[60, 200],
                 dcomplex='hilbert', cycle=(3, 6), width=7, n_bins=18,
                 verbose=None):
        """Check and initialize."""
        set_log_level(verbose)
        self._idcheck(idpac)
        _PacObj.__init__(self, f_pha=f_pha, f_amp=f_amp, dcomplex=dcomplex,
                         cycle=cycle, width=width)
        _PacPlt.__init__(self)
        self.n_bins = int(n_bins)
        logger.info("Phase Amplitude Coupling object defined")

    def fit(self, pha, amp, n_perm=200, p=.05, mcp='maxstat', n_jobs=-1,
            random_state=None, verbose=None):
        """Compute PAC on filtered data.

        Parameters
        ----------
        pha : array_like
            Array of phases of shape (n_pha, n_epochs, n_times).
            Angles should be in rad.
        amp : array_like
            Array of amplitudes of shape (n_amp, n_epochs, n_times).
        n_perm : int | 200
            Number of surrogates to compute.
        p : float | 0.05
            Statistical threshold
        mcp : {'fdr', 'bonferroni'}
            Correct the p-values for multiple comparisons. Use either :

                * 'maxstat' : maximum statistics
                * 'fdr' : FDR correction (need MNE-Python)
                * 'bonferroni' : Bonferroni correction (need MNE-Python)
        n_jobs : int | -1
            Number of jobs to compute PAC in parallel. For very large data,
            set this parameter to 1 in order to prevent large memory usage.
        random_state : int | None
            Fix the random state of the machine for reproducible results.

        Returns
        -------
        pac : array_like
            Phase-Amplitude Coupling measure of shape (n_amp, n_pha, n_epochs)

        Attributes
        ----------
        pac : array_like
            Unormalized Phase-Amplitude Coupling measure of shape (n_amp,
            n_pha, n_epochs)
        pvalues : array_like
            Array of p-values of shape (n_amp, n_pha)
        surrogates : array_like
            Array of surrogates of shape (n_perm, n_amp, n_pha, n_epochs)
        """
        set_log_level(verbose)
        # ---------------------------------------------------------------------
        # input checking
        pha, amp = self._phampcheck(pha, amp)
        self._pvalues, self._surrogates = None, None
        # for the plv, extract the phase of the amplitude
        if self._idpac[0] == 5:
            amp = np.angle(hilbertm(amp))

        # ---------------------------------------------------------------------
        # check if permutations should be computed
        if self._idpac[1] == 0:
            n_perm = None
        if not isinstance(n_perm, int) or not (n_perm > 0):
            self._idpac = (self._idpac[0], 0, 0)
            compute_surro = False
        else:
            compute_surro = True

        # ---------------------------------------------------------------------
        # copnorm if gaussian copula is used
        if self._idpac[0] == 6:
            logger.debug(f"    copnorm the phase and the amplitude")
            pha = copnorm(np.stack([np.sin(pha), np.cos(pha)], axis=-2))
            amp = copnorm(amp[..., np.newaxis, :])

        # ---------------------------------------------------------------------
        # true pac estimation
        logger.info(f'    true PAC estimation using {self.method}')
        fcn = get_pac_fcn(self.idpac[0], self.n_bins, p)
        pac = fcn(pha, amp)
        self._pac = pac.copy()

        # ---------------------------------------------------------------------
        # compute surrogates (if needed)
        if compute_surro:
            if random_state is None:
                random_state = int(np.random.randint(0, 10000, size=1))
            logger.info(f"    compute surrogates ({self.str_surro}, {n_perm} "
                        f"permutations, random_state={random_state})")
            surro = compute_surrogates(pha, amp, self.idpac[1], fcn, n_perm,
                                       n_jobs, random_state)
            self._surrogates = surro

            # infer pvalues
            self.infer_pvalues(p, mcp=mcp)

        # ---------------------------------------------------------------------
        # normalize (if needed)
        if self._idpac[2] != 0:
            # Get the mean / deviation of surrogates
            logger.info("    normalize true PAC estimation by surrogates "
                        f"({self.str_norm})")
            normalize(self.idpac[2], pac, surro)

        return pac

    def filterfit(self, sf, x_pha, x_amp=None, n_perm=200, p=.05,
                  mcp='maxstat', edges=None, n_jobs=-1, random_state=None,
                  verbose=None):
        """Filt the data then compute PAC on it.

        Parameters
        ----------
        sf : float
            The sampling frequency.
        x_pha, x_amp : array_like
            Array of data for computing PAC. x_pha is the data used for
            extracting phases and x_amp, amplitudes. Both arrays must have
            the same shapes. If you want to compute PAC locally i.e. on the
            same electrode, x=x_pha=x_amp. For distant coupling, x_pha and
            x_amp could be different but still must to have the same shape.
        n_perm : int | 200
            Number of surrogates to compute.
        p : float | 0.05
            Statistical threshold
        mcp : {'fdr', 'bonferroni'}
            Correct the p-values for multiple comparisons. Use either :

                * 'maxstat' : maximum statistics
                * 'fdr' : FDR correction (need MNE-Python)
                * 'bonferroni' : Bonferroni correction (need MNE-Python)
        edges : int | None
            Number of samples to discard to avoid edge effects due to filtering
        n_jobs : int | -1
            Number of jobs to compute PAC in parallel. For very large data,
            set this parameter to 1 in order to prevent large memory usage.
        random_state : int | None
            Fix the random state of the machine for reproducible results.

        Returns
        -------
        pac : array_like
            Phase-Amplitude Coupling measure of shape (namp, npha, ...).

        Attributes
        ----------
        pac : array_like
            Unormalized Phase-Amplitude Coupling measure of shape (n_amp,
            n_pha, n_epochs)
        pvalues : array_like
            Array of p-values of shape (n_amp, n_pha)
        surrogates : array_like
            Array of surrogates of shape (n_perm, n_amp, n_pha, n_epochs)
        """
        # Check if amp is None :
        if x_amp is None:
            x_amp = x_pha
        # Shape checking :
        assert x_pha.shape == x_amp.shape, ("Inputs `x_pha` and `x_amp` must "
                                            "have the same shape.")
        # Extract phase (npha, ...) and amplitude (namp, ...) :
        logger.info(f"    extract phases (n_pha={len(self.xvec)}) and "
                    f"amplitudes (n_amps={len(self.yvec)})")
        kw = dict(keepfilt=False, edges=edges, n_jobs=1)
        pha = self.filter(sf, x_pha, 'phase', **kw)
        amp = self.filter(sf, x_amp, 'amplitude', **kw)

        # Special cases :
        if self._idpac[0] == 5:
            amp = np.angle(hilbertm(amp))

        # Compute pac :
        return self.fit(pha, amp, p=p, mcp=mcp, n_perm=n_perm, n_jobs=n_jobs,
                        random_state=random_state, verbose=verbose)

    def infer_pvalues(self, p=0.05, mcp='maxstat'):
        """Infer p-values based on surrogate distribution.

        Parameters
        ----------
        p : float | 0.05
            Significiency threshold

        Returns
        -------
        pvalues : array_like
            Array of p-values of shape (n_amp, n_pha)
        mcp : {'fdr', 'bonferroni'}
            Correct the p-values for multiple comparisons. Use either :

                * 'maxstat' : maximum statistics
                * 'fdr' : FDR correction (need MNE-Python)
                * 'bonferroni' : Bonferroni correction (need MNE-Python)
        """
        # ---------------------------------------------------------------------
        # check that pac and surrogates has already been computed
        assert hasattr(self, 'pac'), ("You should compute PAC first. Use the "
                                      "`fit` method")
        assert hasattr(self, 'surrogates'), "No surrogates computed"

        # mean pac and surrogates across trials
        m_pac, m_surro = self.pac.mean(2), self.surrogates.mean(3)
        self._pvalues = self._infer_pvalues(m_pac, m_surro, p=p, mcp=mcp)

        return self._pvalues

    def _idcheck(self, idpac):
        """Check the idpac parameter."""
        idpac = np.atleast_1d(idpac)
        if not all([isinstance(k, int) for k in idpac]) and (len(idpac) != 3):
            raise ValueError("idpac must be a tuple/list of 3 integers.")
        # Ozkurt PAC case (doesn't need surrogates and normalization)
        if idpac[0] == 4:
            idpac = np.array([4, 0, 0])
        if (idpac[0] == 1) and (idpac[1] == 0) and (idpac[2] == 0):
            logger.warning(
                "MVL is amplitude dependent which means that if the amplitude "
                "increases, MVL also increases. You should select a "
                "normalization method for correcting this limitation "
                "(e.g idpac=(1, 2, 4))")
        if (idpac[2] != 0) and (idpac[1] == 0):
            logger.warning("If you want to normalize the estimated PAC, you "
                           "should select a surrogate method (second digit of "
                           "`idpac`). Normalization ignored.")
            idpac[2] = 0
        self._idpac = idpac
        # string representation
        self.method, self.str_surro, self.str_norm = pacstr(idpac)

    @property
    def idpac(self):
        """Get the idpac value."""
        return self._idpac

    @idpac.setter
    def idpac(self, value):
        """Set idpac value."""
        self._idcheck(value)

    @property
    def pac(self):
        """Array of un-normalized PAC of shape (n_amp, n_pha, n_epochs)."""
        return self._pac

    @property
    def surrogates(self):
        """Array of surrogates of shape (n_perm, n_amp, n_pha, n_epochs)."""
        return self._surrogates

    @property
    def pvalues(self):
        """Array of p-values of shape (n_amp, n_pha)."""
        return self._pvalues


class EventRelatedPac(_PacObj, _PacVisual):
    """Compute the Event Related Phase-Amplitude Coupling (ERPAC).

    The traditional PAC approach is computed across time, hence this means that
    you can't observe PAC changes across time. In contrast, the ERPAC is
    computed across epochs (or trials) which preserves the time dimension.

    Parameters
    ----------
    f_pha, f_amp : list/tuple/array | def: [2, 4] and [60, 200]
        Frequency vector for the phase and amplitude. Here you can use
        several forms to define those vectors :

            * Basic list/tuple (ex: [2, 4] or [8, 12]...)
            * List of frequency bands (ex: [[2, 4], [5, 7]]...)
            * Dynamic definition : (start, stop, width, step)
            * Range definition (ex : np.arange(3) => [[0, 1], [1, 2]])

    dcomplex : {'wavelet', 'hilbert'}
        Method for the complex definition. Use either 'hilbert' or
        'wavelet'.
    cycle : tuple | (3, 6)
        Control the number of cycles for filtering (only if dcomplex is
        'hilbert'). Should be a tuple of integers where the first one
        refers to the number of cycles for the phase and the second for the
        amplitude.
    width : int | 7
        Width of the Morlet's wavelet.
    """

    def __init__(self, f_pha=[2, 4], f_amp=[60, 200], dcomplex='hilbert',
                 cycle=(3, 6), width=7, verbose=None):
        """Check and initialize."""
        set_log_level(verbose)
        _PacObj.__init__(self, f_pha=f_pha, f_amp=f_amp, dcomplex=dcomplex,
                         cycle=cycle, width=width)
        _PacPlt.__init__(self)
        logger.info("Event Related PAC object defined")

    def fit(self, pha, amp, method='circular', smooth=None, n_jobs=-1,
            n_perm=None, p=.05, mcp='fdr', verbose=None):
        """Compute the Event-Related Phase-Amplitude Coupling (ERPAC).

        The ERPAC :cite:`voytek2013method` is used to measure PAC across trials
        and is interesting for real-time estimation.

        Parameters
        ----------
        pha, amp : array_like
            Respectively the phase of slower oscillations of shape
            (n_pha, n_epochs, n_times) and the amplitude of faster
            oscillations of shape (n_pha, n_epochs, n_times).
        method : {'circular', 'gc'}
            Name of the method for computing erpac. Use 'circular' for
            reproducing :cite:`voytek2013method` or 'gc' for a Gaussian-Copula
            based erpac :cite:`ince2017statistical`.
        smooth : int | None
            Half number of time-points to use to produce a smoothing. Only
            active with the Gaussian-Copula ('gc') method.
        n_perm : int | None
            Number of permutations to compute for assessing p-values for the
            gaussian-copula ('gc') method. Statistics are performed by randomly
            swapping phase trials
        p : float | 0.05
            Statistical threshold for the gaussian-copula ('gc') method
        mcp : {'fdr', 'bonferroni'}
            Correct the p-values for multiple comparisons. This is needed when
            using the circular ERPAC (:cite:`voytek2013method`). Note that the
            correction is performed using MNE-Python.

        Returns
        -------
        erpac : array_like
            The ERPAC estimation of shape (n_amp, n_pha, n_times)
        """
        set_log_level(verbose)
        pha, amp = self._phampcheck(pha, amp)
        self.method = method
        self._pvalues = None
        # move the trial axis to the end (n_freqs, n_times, n_epochs)
        pha, amp = np.moveaxis(pha, 1, -1), np.moveaxis(amp, 1, -1)
        # method switch
        if method == 'circular':
            self.method = "ERPAC (Voytek et al. 2013)"
            logger.info(f"    Compute {self.method}")
            self._erpac, self._pvalues = erpac(pha, amp)
            self.infer_pvalues(p=p, mcp=mcp)
        elif method == 'gc':
            self.method = "Gaussian-Copula ERPAC"
            logger.info(f"    Compute {self.method}")
            # copnorm phases and amplitudes then compute erpac
            sco = copnorm(np.stack([np.sin(pha), np.cos(pha)], axis=-2))
            amp = copnorm(amp)[..., np.newaxis, :]
            self._erpac = ergcpac(sco, amp, smooth=smooth, n_jobs=n_jobs)
            # compute permutations (if needed)
            if isinstance(n_perm, int) and (n_perm > 0):
                logger.info(f"    Compute {n_perm} permutations")
                self._surrogates = _ergcpac_perm(sco, amp, smooth=smooth,
                                                 n_jobs=n_jobs, n_perm=n_perm)
                self.infer_pvalues(p=p, mcp=mcp)
        return self.erpac

    def filterfit(self, sf, x_pha, x_amp=None, method='circular', smooth=None,
                  n_perm=None, p=.05, mcp='fdr', edges=None, n_jobs=-1,
                  verbose=None):
        """Extract phases, amplitudes and compute ERPAC.

        Parameters
        ----------
        sf : float
            The sampling frequency.
        x_pha, x_amp : array_like
            Array of data for computing ERPAC. x_pha is the data used for
            extracting phases and x_amp, amplitudes. Both arrays must have
            the same shapes (i.e n_epochs, n_times). If you want to compute
            local ERPAC i.e. on the same electrode, x=x_pha=x_amp. For distant
            coupling, x_pha and x_amp could be different but still must to have
            the same shape.
        method : {'circular', 'gc'}
            Name of the method for computing erpac. Use 'circular' for
            reproducing :cite:`voytek2013method` or 'gc' for a Gaussian-Copula
            based erpac.
        smooth : int | None
            Half number of time-points to use to produce a smoothing. Only
            active with the Gaussian-Copula ('gc') method
            :cite:`ince2017statistical`.
        n_perm : int | None
            Number of permutations to compute for assessing p-values for the
            gaussian-copula ('gc') method. Statistics are performed by randomly
            swapping phase trials
        p : float | 0.05
            Statistical threshold for the gaussian-copula ('gc') method
        mcp : {'fdr', 'bonferroni'}
            Correct the p-values for multiple comparisons. This is needed when
            using the circular ERPAC (:cite:`voytek2013method`). Note that the
            correction is performed using MNE-Python.
        edges : int | None
            Number of samples to discard to avoid edge effects due to filtering

        Returns
        -------
        erpac : array_like
            The ERPAC estimation of shape (n_amp, n_pha, n_times)
        """
        x_amp = x_pha if not isinstance(x_amp, np.ndarray) else x_amp
        # extract phases and amplitudes
        logger.info(f"    Extract phases (n_pha={len(self.xvec)}) and "
                    f"amplitudes (n_amps={len(self.yvec)})")
        kw = dict(keepfilt=False, edges=edges, n_jobs=1)
        pha = self.filter(sf, x_pha, ftype='phase', **kw)
        amp = self.filter(sf, x_amp, ftype='amplitude', **kw)
        # compute erpac
        return self.fit(pha, amp, method=method, smooth=smooth, n_jobs=n_jobs,
                        n_perm=n_perm, p=p, mcp=mcp, verbose=verbose)

    def infer_pvalues(self, p=0.05, mcp='fdr'):
        """Infer p-values based on surrogate distribution.

        Parameters
        ----------
        p : float | 0.05
            Statistical threshold
        mcp : {'fdr', 'bonferroni'}
            Correct the p-values for multiple comparisons. This is needed when
            using the circular ERPAC (:cite:`voytek2013method`). Note that the
            correction is performed using MNE-Python.

        Returns
        -------
        pvalues : array_like
            Array of p-values of shape (n_amp, n_pha, n_times)
        """
        # ---------------------------------------------------------------------
        # check that pac and surrogates has already been computed
        assert hasattr(self, 'erpac'), ("You should compute ERPAC first. Use "
                                        "the `fit` method")
        assert mcp in ['fdr', 'bonferroni']

        # correct the p-values for multiple comparisons (Voytek's only)
        if "Voytek" in self.method:
            logger.info(f"    Correct p-values for multiple-comparisons using "
                        f"{mcp} correction of MNE-Python")
            from mne.stats import fdr_correction, bonferroni_correction
            fcn = fdr_correction if mcp is 'fdr' else bonferroni_correction
            _, self._pvalues = fcn(self._pvalues, alpha=p)
        else:
            assert hasattr(self, 'surrogates'), "No surrogates computed"
            # compute the p-values using maxstat (gcPAC)
            self._pvalues = self._infer_pvalues(self.erpac, self.surrogates,
                                                p=p)

        return self._pvalues

    @property
    def erpac(self):
        """Array of event-related PAC of shape ()."""
        return self._erpac

    @property
    def surrogates(self):
        """Array of surrogates of shape (n_perm, n_amp, n_pha, n_times)."""
        return self._surrogates

    @property
    def pvalues(self):
        """Array of p-values of shape (n_amp, n_pha, n_times)."""
        return self._pvalues


class PreferredPhase(_PacObj, _PolarPlt):
    """Compute the Preferred Phase (PP).

    The preferred phase is defined as the phase at which the amplitude is
    maximum.

    Parameters
    ----------
    f_pha, f_amp : list/tuple/array | def: [2, 4] and [60, 200]
        Frequency vector for the phase and amplitude. Here you can use
        several forms to define those vectors :

            * Basic list/tuple (ex: [2, 4] or [8, 12]...)
            * List of frequency bands (ex: [[2, 4], [5, 7]]...)
            * Dynamic definition : (start, stop, width, step)
            * Range definition (ex : np.arange(3) => [[0, 1], [1, 2]])

    dcomplex : {'wavelet', 'hilbert'}
        Method for the complex definition. Use either 'hilbert' or
        'wavelet'.
    cycle : tuple | (3, 6)
        Control the number of cycles for filtering (only if dcomplex is
        'hilbert'). Should be a tuple of integers where the first one
        refers to the number of cycles for the phase and the second for the
        amplitude.
    width : int | 7
        Width of the Morlet's wavelet.
    """

    def __init__(self, f_pha=[2, 4], f_amp=[60, 200], dcomplex='hilbert',
                 cycle=(3, 6), width=7, verbose=None):
        """Check and initialize."""
        set_log_level(verbose)
        _PacObj.__init__(self, f_pha=f_pha, f_amp=f_amp, dcomplex=dcomplex,
                         cycle=cycle, width=width)
        _PacPlt.__init__(self)
        logger.info("Preferred phase object defined")
        self.method = 'Preferred-Phase (PP)'

    def fit(self, pha, amp, n_bins=72):
        """Compute the preferred-phase.

        Parameters
        ----------
        pha, amp : array_like
            Respectively the phase of slower oscillations of shape
            (n_pha, n_epochs, n_times) and the amplitude of faster
            oscillations of shape (n_pha, n_epochs, n_times).
        n_bins : int | 72
            Number of bins for bining the amplitude according to phase
            slices.

        Returns
        -------
        binned_amp : array_like
            The binned amplitude according to the phase of shape
            (n_bins, n_amp, n_pha, n_epochs)
        pp : array_like
            The prefered phase where the amplitude is maximum of shape
            (namp, npha, n_epochs)
        polarvec : array_like
            The phase vector for the polar plot of shape (n_bins,)
        """
        # Check phase and amplitude shapes :
        pha, amp = self._phampcheck(pha, amp)
        return preferred_phase(pha, amp, n_bins=n_bins)

    def filterfit(self, sf, x_pha, x_amp=None, edges=None, n_bins=12,
                  verbose=None):
        """Extract phases, amplitudes and compute the preferred phase (PP).

        Parameters
        ----------
        sf : float
            The sampling frequency.
        x_pha, x_amp : array_like
            Array of data for computing PP. x_pha is the data used for
            extracting phases and x_amp, amplitudes. Both arrays must have
            the same shapes (i.e n_epochs, n_times). If you want to compute
            local PP i.e. on the same electrode, x=x_pha=x_amp. For distant
            coupling, x_pha and x_amp could be different but still must to have
            the same shape.
        n_bins : int | 72
            Number of bins for bining the amplitude according to phase
            slices.
        edges : int | None
            Number of samples to discard to avoid edge effects due to filtering

        Returns
        -------
        binned_amp : array_like
            The binned amplitude according to the phase of shape
            (n_bins, n_amp, n_pha, n_epochs)
        pp : array_like
            The prefered phase where the amplitude is maximum of shape
            (namp, npha, n_epochs)
        polarvec : array_like
            The phase vector for the polar plot of shape (n_bins,)
        """
        x_amp = x_pha if not isinstance(x_amp, np.ndarray) else x_amp
        # extract phases and amplitudes
        logger.info(f"    Extract phases (n_pha={len(self.xvec)}) and "
                    f"amplitudes (n_amps={len(self.yvec)})")
        kw = dict(keepfilt=False, edges=edges, n_jobs=1)
        pha = self.filter(sf, x_pha, ftype='phase', **kw)
        amp = self.filter(sf, x_amp, ftype='amplitude', **kw)
        # compute pp
        return self.fit(pha, amp, n_bins=n_bins)
