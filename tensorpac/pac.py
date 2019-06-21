"""Main PAC class."""
import numpy as np
import logging

from tensorpac.spectral import spectral, hilbertm
from tensorpac.methods import (get_pac_fcn, _kl_hr, pacstr, compute_surrogates,
                               normalize)
from tensorpac.gcmi import nd_mi_gg, copnorm
from tensorpac.stats import circ_corrcc
from tensorpac.utils import pac_vec
from tensorpac.visu import PacPlot
from tensorpac.io import set_log_level
from tensorpac.config import MNE_EPOCHS_TYPE

logger = logging.getLogger('tensorpac')


class Pac(PacPlot):
    """Compute Phase-Amplitude Coupling (PAC) using tensors.

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

            - '1' : Mean Vector Length (MVL) [#f1]_
            - '2' : Kullback-Leibler Distance (KLD) [#f2]_
            - '3' : Heights Ratio (HR) [#f3]_
            - '4' : ndPAC [#f4]_
            - '5' : Phase Synchrony [#f3]_
            - '6' : Gaussian Copula PAC

        * Second digit : refer to the method for computing surrogates

            - '0' : No surrogates
            - '1' : Swap phase / amplitude across trials [#f2]_
            - '2' : Swap amplitude time blocks [#f5]_
            - '3' : Time lag [#f1]_

        * Third digit : refer to the normalization method for correction

            - '0' : No normalization
            - '1' : Substract the mean of surrogates
            - '2' : Divide by the mean of surrogates
            - '3' : Substract then divide by the mean of surrogates
            - '4' : Z-score

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
    filt : {'fir1', 'butter', 'bessel'}
        Filtering method (only if dcomplex is 'hilbert'). Choose either
        'fir1', 'butter' or 'bessel'
    cycle : tuple | (3, 6)
        Control the number of cycles for filtering (only if dcomplex is
        'hilbert'). Should be a tuple of integers where the first one
        refers to the number of cycles for the phase and the second for the
        amplitude [#f5]_.
    filtorder : int | 3
        Filter order for the Butterworth and Bessel filters (only if
        dcomplex is 'hilbert').
    width : int | 7
        Width of the Morlet's wavelet.
    n_bins : int | 18
        Number of bins for the KLD and HR PAC method [#f2]_ [#f3]_

    .. warning::

        * The ndPac [#f4]_ include a fast and reliable statistical test. As a
          result, if the ndPAC is choosed as the main PAC method, surrogates
          and normalization will be deactivate.

    Attributes
    ----------
    xvec, yvec : array_like
        The x and y-vectors for plotting.
    f_pha, f_amp : array_like
        The phase and amplitude frequency vectors for pac extraction.

    References
    ----------
    .. rubric:: Footnotes
    .. [#f1] `Canolty et al, 2006 <http://www.ncbi.nlm.nih.gov/pmc/articles/
       PMC2628289/>`_
    .. [#f2] `Tort et al, 2010 <http://www.ncbi.nlm.nih.gov/pmc/articles/
       PMC2941206/>`_
    .. [#f3] `Lakatos et al, 2005 <https://www.ncbi.nlm.nih.gov/pubmed/
       15901760>`_
    .. [#f4] `Ozkurt et al, 2012 <http://www.ncbi.nlm.nih.gov/pubmed/
       22531738/>`_
    .. [#f5] `Bahramisharif et al, 2013 <http://www.jneurosci.org/content/33/
       48/18849.short/>`_
    """

    ###########################################################################
    #                              __FCN__
    ###########################################################################
    def __init__(self, idpac=(1, 2, 3), f_pha=[2, 4], f_amp=[60, 200],
                 dcomplex='hilbert', filt='fir1', cycle=(3, 6), filtorder=3,
                 width=7, n_bins=18, verbose=None):
        """Check and initialize."""
        set_log_level(verbose)
        # Initialize visualization methods :
        PacPlot.__init__(self)
        # ----------------- CHECKING -----------------
        # Pac methods :
        self._idcheck(idpac)
        # Frequency checking :
        self.f_pha, self.f_amp = pac_vec(f_pha, f_amp)
        self.xvec, self.yvec = self.f_pha.mean(1), self.f_amp.mean(1)
        self.n_bins = int(n_bins)

        # Check spectral properties :
        self._speccheck(filt, dcomplex, filtorder, cycle, width)

    def __str__(self):
        """String representation."""
        return self.method

    ###########################################################################
    #                              METHODS
    ###########################################################################
    def filter(self, sf, x, ftype='phase', keepfilt=False, n_jobs=-1):
        """Filt the data in the specified frequency bands.

        Parameters
        ----------
        sf: float
            The sampling frequency.
        x: array_like
            Array of data of shape (n_trials, n_channels, n_pts)
        ftype : {'phase', 'amplitude'}
            Specify if you want to extract phase ('phase') or the amplitude
            ('amplitude').
        n_jobs : int | -1
            Number of jobs to compute PAC in parallel. For very large data,
            set this parameter to 1 in order to prevent large memory usage.
        keepfilt : bool | False
            Specify if you only want the filtered data (True). This parameter
            is only available with dcomplex='hilbert' and not wavelet.

        Returns
        -------
        xfilt : array_like
            The filtered data of shape (n_freqs, n_trials, n_channels, n_pts)
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
        if not isinstance(x, np.ndarray) and type(x) in MNE_EPOCHS_TYPE:
            x = x.get_data()
            sf = x.info['sfreq']
        assert x.ndim == 3, ("x should be a 3d array like (n_trials, "
                             "n_channels, n_pts)")

        # ---------------------------------------------------------------------
        # Switch between phase or amplitude :
        if ftype is 'phase':
            tosend = 'pha' if not keepfilt else None
            xfilt = spectral(x, sf, self.f_pha, tosend, self._dcomplex,
                             self._filt, self._filtorder, self._cycle[0],
                             self._width, n_jobs)
        elif ftype is 'amplitude':
            tosend = 'amp' if not keepfilt else None
            xfilt = spectral(x, sf, self.f_amp, tosend, self._dcomplex,
                             self._filt, self._filtorder, self._cycle[1],
                             self._width, n_jobs)
        return xfilt

    def fit(self, pha, amp, n_perm=200, p=.05, n_jobs=-1, verbose=None):
        """Compute PAC on filtered data.

        Parameters
        ----------
        pha : array_like
            Array of phases of shape (n_pha, n_trials, n_channels, n_pts).
            Angles should be in rad.
        amp : array_like
            Array of amplitudes of shape (n_amp, n_trials, n_channels, n_pts).
        n_perm : int | 200
            Number of surrogates to compute.
        p : float | 0.05
            Statistical threhold
        n_jobs : int | -1
            Number of jobs to compute PAC in parallel. For very large data,
            set this parameter to 1 in order to prevent large memory usage.

        Returns
        -------
        pac: array_like
            Phase-Amplitude Coupling measure of shape (n_amp, n_pha, n_trials,
            n_channels)

        Attributes
        ----------
        pvalues_ : array_like
            Array of p-values of shape (n_amp, n_pha, n_channels)
        surrogates_ : array_like
            Array of surrogates of shape (n_perm, n_amp, n_pha, n_trials,
            n_channels)
        """
        set_log_level(verbose)
        # ---------------------------------------------------------------------
        # input checking
        pha, amp = self._phampcheck(pha, amp)
        self.pvalues_, self.surrogates_ = None, None
        # for the phase synchrony, extract the phase of the amplitude
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
            logger.info(f"    copnorm the phase and the amplitude")
            pha = copnorm(np.stack([np.sin(pha), np.cos(pha)], axis=-2))
            amp = copnorm(amp[..., np.newaxis, :])

        # ---------------------------------------------------------------------
        # true pac estimation
        logger.info(f'    true PAC estimation using {self.method}')
        fcn = get_pac_fcn(self.idpac[0], self.n_bins, p)
        pac = fcn(pha, amp)
        self.pac_ = pac

        # ---------------------------------------------------------------------
        # compute surrogates (if needed)
        if compute_surro:
            logger.info(f"    compute surrogates ({self.str_surro}, {n_perm} "
                        "permutations)")
            suro = compute_surrogates(pha, amp, self.idpac[1], fcn, n_perm,
                                      n_jobs)
            self.surrogates_ = suro

            # infer pvalues
            self.infer_pvalues(p)

        # ---------------------------------------------------------------------
        # normalize (if needed)
        if self._idpac[2] != 0:
            # Get the mean / deviation of surrogates
            m_surro, std_surro = np.mean(suro, axis=0), np.std(suro, axis=0)
            logger.info("    normalize true PAC estimation by surrogates "
                        f"({self.str_norm})")
            normalize(pac, m_surro, std_surro, self.idpac[2])

        return pac

    def filterfit(self, sf, x_pha, x_amp=None, n_perm=200, p=.05, n_jobs=-1,
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
            Statistical threhold
        n_jobs : int | -1
            Number of jobs to compute PAC in parallel. For very large data,
            set this parameter to 1 in order to prevent large memory usage.

        Returns
        -------
        pac: array_like
            Phase-Amplitude Coupling measure of shape (namp, npha, ...).
        """
        # Check if amp is None :
        if x_amp is None:
            x_amp = x_pha
        # Shape checking :
        assert x_pha.shape == x_amp.shape, ("Inputs `x_pha` and `x_amp` must "
                                            "have the same shape.")
        # Extract phase (npha, ...) and amplitude (namp, ...) :
        logger.info(f"    Extract phases (n_pha={len(self.xvec)}) and "
                    f"amplitudes (n_amps={len(self.yvec)})")
        pha = self.filter(sf, x_pha, 'phase', False, n_jobs)
        amp = self.filter(sf, x_amp, 'amplitude', False, n_jobs)

        # Special cases :
        if self._idpac[0] == 5:
            amp = np.angle(hilbertm(amp))

        # Compute pac :
        return self.fit(pha, amp, p=p, n_perm=n_perm, n_jobs=n_jobs,
                        verbose=verbose)

    def infer_pvalues(self, p=0.05):
        """Infer p-values based on surrogate distribution.

        Parameters
        ----------
        p : float | 0.05
            Statistical threhold

        Returns
        -------
        pvalues : array_like
            Array of p-values of shape (n_amp, n_pha, n_channels)
        """
        # ---------------------------------------------------------------------
        # check that pac and surrogates has already been computed
        assert hasattr(self, 'pac_'), ("You should compute PAC first. Use the "
                                       "`fit` method")
        assert hasattr(self, 'surrogates_'), "No surrogates computed"
        assert all([isinstance(k, np.ndarray) for k in (
            self.pac_, self.surrogates_)])
        n_perm = self.surrogates_.shape[0]

        # ---------------------------------------------------------------------
        # mean pac and surrogates across trials
        m_pac, m_surro = self.pac_.mean(2), self.surrogates_.mean(3)
        self.pvalues_ = np.ones_like(m_pac)
        # infer pvalues
        logger.info(f"    infer p-values at p={p}")
        max_dist = m_surro.reshape(n_perm, -1).max(1)
        th = np.percentile(max_dist, 100. * (1 - p), axis=0,
                           interpolation='nearest')
        self.pvalues_[m_pac > th] = p
        return self.pvalues_

    def pp(self, pha, amp, n_bins=72):
        """Compute the preferred-phase.

        Parameters
        ----------
        pha : array_like
            Phase of slower oscillations.
        amp : array_like
            Amplitude of fastest oscillations.
        n_bins : int | 72
            Number of bins for bining the amplitude according to phase
            slices.

        Returns
        -------
        ampbin : array_like
            The binned amplitude according to the phase of shape
            (n_bins, namp, npha...).
        pp : array_like
            The prefered phase where the amplitude is maximum of shape
            (namp, npha, ...).
        polarvec : array_like
            The phase vector for the polar plot.
        """
        # Check phase and amplitude shapes :
        pha, amp = self._phampcheck(pha, amp)
        # Define the method name :
        self.method = 'Preferred-Phase (PP)' 
        self.str_surro, self.str_norm = '', ''
        # Bin the amplitude according to the phase :
        ampbin = _kl_hr(pha, amp, n_bins)
        ampbin /= ampbin.sum(axis=0, keepdims=True)
        # Find the index where the amplitude is maximum over the bins :
        idxmax = ampbin.argmax(axis=0)
        # Find the preferred phase :
        binsize = (2 * np.pi) / float(n_bins)
        vecbin = np.arange(-np.pi, np.pi, binsize) + binsize / 2
        pp = vecbin[idxmax]
        # Build the phase vector (polar plot) :
        polarvec = np.linspace(-np.pi, np.pi, ampbin.shape[0])
        return ampbin, pp, polarvec

    def erpac(self, pha, amp, method='circular', verbose=None):
        """Compute the Event-Related Phase-Amplitude Coupling (ERPAC).

        The ERPAC [#f6]_ is used to measure PAC across trials and is
        interesting for real-time estimation.

        Parameters
        ----------
        pha, amp : array_like
            Respectively the phase of slower oscillations of shape
            (n_pha, n_trials, n_channels, n_pts) and the amplitude of faster
            oscillations of shape (n_pha, n_trials, n_channels, n_pts).
        method : {'circular', 'gc'}
            Name of the method for computing erpac. Use 'circular' for
            reproducing [#f6]_ or 'gc' for a Gaussian-Copula based erpac.

        Returns
        -------
        erpac : array_like
            The ERPAC estimation.

        Attributes
        ----------
        pvalues_ : array_like
            The associated p-values (only if method='circular')

        .. [#f6] `Voytek et al, 2013 <https://www.ncbi.nlm.nih.gov/pubmed/
           22986076>`_
        """
        set_log_level(verbose)
        pha, amp = self._phampcheck(pha, amp)
        self.str_surro, self.str_norm = '', ''
        # Move the trial axis to the end :
        pha = np.moveaxis(pha, 1, -1)
        amp = np.moveaxis(amp, 1, -1)
        # method switch
        if method == 'circular':
            self.method = "ERPAC (Voytek et al. 2013)"
            logger.info(f"Compute {self.method}")
            erpac, self.pvalues_ = circ_corrcc(pha, amp)
        elif method == 'gc':
            self.method = "Gaussian-Copula ERPAC"
            logger.info(f"Compute {self.method}")
            # get shapes
            n_pha, n_chans, n_pts, n_trials = pha.shape
            n_amp = amp.shape[0]
            # conversion for computing mi
            sco = copnorm(np.stack([np.sin(pha), np.cos(pha)], axis=-2))
            amp = copnorm(amp)[..., np.newaxis, :]
            erpac = np.zeros((n_amp, n_pha, n_chans, n_pts))
            for a in range(n_amp):
                for p in range(n_pha):
                    erpac[a, p, ...] = nd_mi_gg(sco[p, ...], amp[a, ...],
                                                mvaxis=-2, traxis=-1,
                                                biascorrect=False)
            self.pvalues_ = None
        return erpac

    ###########################################################################
    #                              CHECKING
    ###########################################################################
    def _idcheck(self, idpac):
        """Check the idpac parameter."""
        idpac = np.atleast_1d(idpac)
        if not all([isinstance(k, int) for k in idpac]) and (len(idpac) != 3):
            raise ValueError("idpac must be a tuple/list of 3 integers.")
        # Ozkurt PAC case (doesn't need surrogates and normalization)
        if idpac[0] == 4:
            idpac = np.array([4, 0, 0])
        if (idpac[2] != 0) and (idpac[1] == 0):
            logger.warning("If you want to normalize the estimated PAC, you "
                           "should select a surrogate method (second digit of "
                           "`idpac`). Normalization ignored.")
            idpac[2] = 0
        self._idpac = idpac
        # string representation
        self.method, self.str_surro, self.str_norm = pacstr(idpac)

    def _speccheck(self, filt=None, dcomplex=None, filtorder=None, cycle=None,
                   width=None):
        """Check spectral parameters."""
        # Check the filter name :
        if filt is not None:
            if filt not in ['fir1', 'butter', 'bessel', 'mne']:
                raise ValueError("filt must either be 'fir1', 'butter' or "
                                 "'bessel'")
            else:
                self._filt = filt
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
        # Convert filtorder :
        if filtorder is not None:
            self._filtorder = int(filtorder)
        # Convert Morlet's width :
        if width is not None:
            self._width = int(width)

    @staticmethod
    def _phampcheck(pha, amp):
        """Check phase and amplitude values."""
        assert pha.ndim == 4, ("`pha` should have a shape of (n_pha, n_trials,"
                               " n_channels, n_pts)")
        assert amp.ndim == 4, ("`amp` should have a shape of (n_pha, n_trials,"
                               " n_channels, n_pts)")
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

    ###########################################################################
    #                              PROPERTIES
    ###########################################################################
    # ----------- IDPAC -----------
    @property
    def idpac(self):
        """Get the idpac value."""
        return self._idpac

    @idpac.setter
    def idpac(self, value):
        """Set idpac value."""
        self._idcheck(value)

    # ----------- FILT -----------
    @property
    def filt(self):
        """Get the filt value."""
        return self._filt

    @filt.setter
    def filt(self, value):
        """Set filt value."""
        self._speccheck(filt=value)

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

    # ----------- FILTORDER -----------
    @property
    def filtorder(self):
        """Get the filtorder value."""
        return self._filtorder

    @filtorder.setter
    def filtorder(self, value):
        """Set filtorder value."""
        self._speccheck(filtorder=value)

    # ----------- WIDTH -----------
    @property
    def width(self):
        """Get the width value."""
        return self._width

    @width.setter
    def width(self, value):
        """Set width value."""
        self._width = value
