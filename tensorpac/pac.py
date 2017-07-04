"""Main PAC class."""
import numpy as np
from scipy.signal import hilbert

from .utils import pac_vec
from .pacstr import pacstr
from .spectral import spectral
from .methods import compute_pac, _kl_hr
from .surrogates import compute_surrogates
from .normalize import normalize
from .visu import PacPlot
from .stats import circ_corrcc


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

        * Second digit : refer to the method for computing surrogates

            - '0' : No surrogates
            - '1' : Swap phase/amplitude across trials [#f2]_
            - '2' : Swap amplitude time blocks [#f5]_
            - '3' : Shuffle amplitude time-series
            - '4' : Time lag [#f1]_

        * Third digit : refer to the normalization method for correction

            - '0' : No normalization
            - '1' : Substract the mean of surrogates
            - '2' : Divide by the mean of surrogates
            - '3' : Substract then divide by the mean of surrogates
            - '4' : Z-score

    fpha, famp : list/tuple/array | def: [2, 4] and [60, 200]
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

    nbins : int | 18
        Number of bins for the KLD and HR PAC method [#f2]_ [#f3]_

    nblocks : int | 2
        Number of blocks for splitting the amplitude. Only active is
        the surrogate method is 2 [#f5]_.

    .. warning::
        * The ndPac [#f4]_ include a fast and reliable statistical test. As a
          result, if the ndPAC is choosed as the main PAC method, surrogates
          and normalization will be deactivate.

        * The phase in a particular frequency band can either be extracted
          using wavelet convolution or filtering followed by the Hilbert
          transform. As a result, every filtering related input (cycle, filt,
          filtorder) are going to be active if the complex decomposition is
          Hilbert.

    Attributes
    ----------
    xvec, yvec : array_like
        The x and y-vectors for plotting.

    fpha, famp : array_like
        The phase and amplitude frequency vectors for pac extraction.

    Methods
    -------
    filt()
        Filt the data in the specified frequency bands.

    fit()
        Run the PAC on filtered data.

    filtfit()
        Filt the data then compute PAC on it.

    pp()
        compute the preferred phase

    erpac()
        compute the Event Related Pac

    pacplot()
        Plotting method that can more generally be used to plot any 2D array.

    comodulogram()
        Plotting method dedicated to the comodulogram.

    triplot()
        Plotting method for the optimal bandwidth search.

    polar()
        Plotting function for circular preferred-phase visualization.

    show()
        Show the figure (equivalent to plt.show())

    savefig()
        Save the figure (equivalent to plt.savefig())

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
    def __init__(self, idpac=(1, 1, 3), fpha=[2, 4], famp=[60, 200],
                 dcomplex='hilbert', filt='fir1', cycle=(3, 6), filtorder=3,
                 width=7, nbins=18, nblocks=2):
        """Check and initialize."""
        # Initialize visualization methods :
        PacPlot.__init__(self)
        # ----------------- CHECKING -----------------
        # Pac methods :
        self._idcheck(idpac)
        # Frequency checking :
        self.fpha, self.famp = pac_vec(fpha, famp)
        self.xvec, self.yvec = self.fpha.mean(1), self.famp.mean(1)

        # Check spectral properties :
        self._speccheck(filt, dcomplex, filtorder, cycle, width)

        # ----------------- SELF -----------------
        self.nbins, self.nblocks = int(nbins), int(nblocks)

    def __str__(self):
        """String representation."""
        st = self.method
        st += '\n' + self.surro if self.surro else ''
        st += '\n' + self.norm if self.norm else ''
        return st

    ###########################################################################
    #                              METHODS
    ###########################################################################
    def filter(self, sf, x, axis=-1, ftype='phase', keepfilt=False, njobs=-1):
        """Filt the data in the specified frequency bands.

        Parameters
        ----------
        sf: float
            The sampling frequency.

        x: array_like
            Array of data.

        axis : int | -1
            Location of the time axis.

        ftype : {'phase', 'amplitude'}
            Specify if you want to extract phase ('phase') or the amplitude
            ('amplitude').

        njobs : int | -1
            Number of jobs to compute PAC in parallel. For very large data,
            set this parameter to 1 in order to prevent large memory usage.

        keepfilt : bool | False
            Specify if you only want the filtered data (True). This parameter
            is only avaible with dcomplex='hilbert' and not wavelet.

        Returns
        -------
        xfilt : array_like
            The filtered data of shape (n_frequency, ...)
        """
        # Sampling frequency :
        if not isinstance(sf, (int, float)):
            raise ValueError("The sampling frequency must be a float number.")
        else:
            sf = float(sf)
        # Compatibility between keepfilt and wavelet :
        if (keepfilt is True) and (self._dcomplex is 'wavelet'):
            raise ValueError("Using wavelet for the complex decomposition do "
                             "not allow to get filtered data only. Set the "
                             "keepfilt parameter to False or set dcomplex to "
                             "'hilbert'.")
        # 1D signals :
        if x.ndim == 1:
            x = x.reshape(1, -1)
            axis = 1
        # Switch between phase or amplitude :
        if ftype is 'phase':
            tosend = 'pha' if not keepfilt else None
            xfilt = spectral(x, sf, self.fpha, axis, tosend, self._dcomplex,
                             self._filt, self._filtorder, self._cycle[0],
                             self._width, njobs)
        elif ftype is 'amplitude':
            tosend = 'amp' if not keepfilt else None
            xfilt = spectral(x, sf, self.famp, axis, tosend, self._dcomplex,
                             self._filt, self._filtorder, self._cycle[1],
                             self._width, njobs)
        else:
            raise ValueError("ftype must either be None, 'phase' or "
                             "'amplitude.'")
        return xfilt

    def fit(self, pha, amp, axis=1, traxis=0, nperm=200, optimize=True,
            get_pval=False, get_surro=False, njobs=-1):
        """Compute PAC on filtered data.

        Parameters
        ----------
        pha, amp: array_like
            Array of filtered data with respectively a shape of (npha, ...)
            and (namp, ...). If you want to compute PAC locally i.e. on the
            same electrode, x=pha=amp. For distant coupling, pha and
            amp could be different but still must to have the same shape.

        axis : int | 1
            Dimension where is located the time axis. By default, the axis
            will be consider as well.

        traxis : int | 0
            Dimension where is located the trial axis. By default the next-
            to-last axis is consider as the trial axis.

        nperm : int | 200
            Number of surrogates to compute.

        optimize : {True, False, 'greedy', 'optimal'}
            Optimize argument of the np.einsum function. Use either False,
            True, 'greedy' or 'optimal'.

        get_pval : bool | False
            Get the pvalues. Only avaible if surrogates are computed.

        get_surro : bool | False
            Return surrogate chance distribution.

        njobs : int | -1
            Number of jobs to compute PAC in parallel. For very large data,
            set this parameter to 1 in order to prevent large memory usage.

        Returns
        -------
        pac: array_like
            Phase-Amplitude Coupling measure of shape (namp, npha, ...).

        pvalue: array_like
            P-values of shape (namp, npha, ...) if get_pval is True.

        suro: array_like
            If get_suro is True, get the chance distribution of shape
            (nperm, namp, npha, ...)

        .. warning::
            * Surrogates are only going to be computed if the second and third
              digits are no 0.

            * The ndPAC use a p value and every non-significant PAC estimation
              is set to zero. This p value is computed as 1/nperm.

            * The traxis argument is only used if you picked up the surrogates
              method 1: "swap phase and amplitude trials [#f2]_"

            * Basically, the surrogate evaluation proposed by [#f5]_ split the
              amplitude into two equal parts, then swap those two blocks. But
              the nblocks parameter allow to split into a larger number.
        """
        # Check phase and amplitude :
        pha, amp, axis = self._phampcheck(pha, amp, axis)
        # For the phase synchrony, extract the phase of the amplitude :
        if self._idpac[0] == 5:
            amp = np.angle(hilbert(amp, axis=axis))
        suro = pvalues = np.array([1.])
        # Check for 0 permutations :
        if nperm in [0, None]:
            self._idpac = (self._idpac[0], 0, 0)
            nperm = 1
            self._csuro = False
        # Compute pac :
        pacargs = (self.idpac[0], self.nbins, 1 / nperm, optimize)
        pac = compute_pac(pha, amp, *pacargs)

        # Compute surrogates (if needed) :
        if self._csuro:
            surargs = (self.idpac[1], axis, traxis, self.nblocks)
            suro = compute_surrogates(pha, amp, surargs, pacargs, nperm, njobs)

            # Get the mean / deviation of surrogates :
            m_surro, std_surro = np.mean(suro, axis=0), np.std(suro, axis=0)

            # Compute statistics :
            suro -= pac[np.newaxis, ...]
            suro.sort(0)
            pvalues = 1. - np.sum(suro < 0., axis=0) / nperm
            pvalues[pvalues < 1 / nperm] = 1 / nperm

            # Normalize pac by surrogates :
            normalize(pac, m_surro, std_surro, self.idpac[2])

        if self._idpac[0] == 4:
            pvalues = np.ones_like(pac)
            pvalues[np.nonzero(pac)] = 1 / nperm

        if not get_surro and not get_pval:
            return pac
        else:
            args = [pac]
            if get_pval and self._csuro:
                args.append(pvalues)
            if get_surro and self._csuro:
                args.append(suro)
            return tuple(args)

    def filterfit(self, sf, xpha, xamp=None, axis=1, traxis=0, nperm=200,
                  optimize=True, get_pval=False, get_surro=False, njobs=-1):
        """Filt the data then compute PAC on it.

        Parameters
        ----------
        sf : float
            The sampling frequency.

        xpha, xamp : array_like
            Array of data for computing PAC. xpha is the data used for
            extracting phases and xamp, amplitudes. Both arrays must have
            the same shapes. If you want to compute PAC locally i.e. on the
            same electrode, x=xpha=xamp. For distant coupling, xpha and
            xamp could be different but still must to have the same shape.

        axis : int | 1
            Dimension where is located the time axis. By default, the axis
            will be consider as well.

        traxis : int | 0
            Dimension where is located the trial axis. By default the next-
            to-last axis is consider as the trial axis.

        nperm : int | 200
            Number of surrogates to compute.

        optimize : {True, False, 'greedy', 'optimal'}
            Optimize argument of the np.einsum function. Use either False,
            True, 'greedy' or 'optimal'.

        get_pval : bool | False
            Get the pvalues. Only avaible if surrogates are computed.

        get_surro : bool | False
            Return surrogate chance distribution.

        njobs : int | -1
            Number of jobs to compute PAC in parallel. For very large data,
            set this parameter to 1 in order to prevent large memory usage.

        Returns
        -------
        pac: array_like
            Phase-Amplitude Coupling measure of shape (namp, npha, ...).

        pvalue: array_like
            P-values of shape (namp, npha, ...) if get_pval is True.

        suro: array_like
            If get_suro is True, get the chance distribution of shape
            (nperm, namp, npha, ...)

        .. warning::
            * Surrogates are only going to be computed if the second and third
              digits are no 0.

            * The ndPAC use a p value and every non-significant PAC estimation
              is set to zero. This p value is computed as 1/nperm.

            * The traxis argument is only used if you picked up the surrogates
              method 1: "swap phase and amplitude trials [#f2]_"

            * Basically, the surrogate evaluation proposed by [#f5]_ split the
              amplitude into two equal parts, then swap those two blocks. But
              the nblocks parameter allow to split into a larger number.
        """
        # Check if amp is None :
        if xamp is None:
            xamp = xpha
        # Shape checking :
        if xpha.shape != xamp.shape:
            raise ValueError("The shape of xpha and xamp must be equals.")
        # Extract phase (npha, ...) and amplitude (namp, ...) :
        pha = self.filter(sf, xpha, axis, 'phase', False, njobs)
        amp = self.filter(sf, xamp, axis, 'amplitude', False, njobs)

        # Special cases :
        if self._idpac[0] == 5:
            amp = np.angle(hilbert(amp, axis=-1))

        # Compute pac :
        return self.fit(pha, amp, axis + 1, traxis + 1, nperm, optimize,
                        get_pval, get_surro, njobs)

    def pp(self, pha, amp, axis=-1, nbins=72, optimize=True):
        """Compute the preferred-phase.

        Parameters
        ----------
        pha : array_like
            Phase of slower oscillations.

        amp : array_like
            Amplitude of fastest oscillations.

        axis : int | -1
            Location of the time axis.

        nbins : int | 72
            Number of bins for bining the amplitude according to phase
            slices.

        optimize : {True, False, 'greedy', 'optimal'}
            Optimize argument of the np.einsum function. Use either False,
            True, 'greedy' or 'optimal'.

        Returns
        -------
        ampbin : array_like
            The binned amplitude according to the phase of shape
            (nbins, namp, npha...).

        pp : array_like
            The prefered phase where the amplitude is maximum of shape
            (namp, npha, ...).

        polarvec : array_like
            The phase vector for the polar plot.
        """
        # Check phase and amplitude shapes :
        pha, amp, axis = self._phampcheck(pha, amp, axis)
        # Define the method name :
        self.method, self.surro, self.norm = 'Preferred-Phase (PP)', '', ''
        # Move the time axis to the end :
        pha = np.moveaxis(pha, axis, -1)
        amp = np.moveaxis(amp, axis, -1)
        # Bin the amplitude according to the phase :
        ampbin = _kl_hr(pha, amp, nbins, optimize)
        ampbin /= ampbin.sum(axis=0, keepdims=True)
        # Find the index where the amplitude is maximum over the bins :
        idxmax = ampbin.argmax(axis=0)
        # Find the preferred phase :
        binsize = (2 * np.pi) / float(nbins)
        vecbin = np.arange(-np.pi, np.pi, binsize) + binsize / 2
        pp = vecbin[idxmax]
        # Build the phase vector (polar plot) :
        polarvec = np.linspace(-np.pi, np.pi, ampbin.shape[0])
        return ampbin, pp, polarvec

    def erpac(self, pha, amp, traxis=0, optimize=True):
        """Compute the Event-Related Phase-Amplitude Coupling (ERPAC).

        The ERPAC [#f6]_ is used to measure PAC across trials and is
        interesting for real-time estimation.

        Parameters
        ----------
        pha : array_like
            Phase of slower oscillations.

        amp : array_like
            Amplitude of fastest oscillations.

        traxis : int | 0
            Location of the trial axis.

        optimize : {True, False, 'greedy', 'optimal'}
            Optimize argument of the np.einsum function. Use either False,
            True, 'greedy' or 'optimal'.

        Returns
        -------
        erpac : array_like
            The ERPAC estimation.

        pvalue : array_like
            The associated p-values.

        .. warning::
            ERPAC is computed across trials, therefor, it does not use an
            *axis* variable but instead, a **traxis** variable which specify
            where is located the axis to consider as trials.

        .. [#f6] `Voytek et al, 2013 <https://www.ncbi.nlm.nih.gov/pubmed/
           22986076>`_
        """
        # Check the phase/amplitude :
        if (pha.ndim <= 1) or (amp.ndim <= 1):
            raise ValueError("The phase and amplitude must have at least two"
                             " dimensions (trials, time).")
        pha, amp, _ = self._phampcheck(pha, amp, traxis)
        # Get method name :
        self.method = "Event-Related Phase-Amplitude Coupling (ERPAC, " + \
                      "Voytek et al. 2013)"
        self.surro, self.norm = '', ''
        # Move the trial axis to the end :
        pha = np.swapaxes(pha, traxis, -1)
        amp = np.swapaxes(amp, traxis, -1)
        # Compute the correlation between the circular phase and le linear
        # amplitude :
        return circ_corrcc(pha, amp, optimize=optimize)

    ###########################################################################
    #                              CHECKING
    ###########################################################################
    def _idcheck(self, idpac):
        """Check the idpac parameter."""
        idpac = np.atleast_1d(idpac)
        self._csuro = True
        if not all([isinstance(k, int) for k in idpac]) and (len(idpac) != 3):
            raise ValueError("idpac must be a tuple/list of 3 integers.")
        else:
            # Ozkurt PAC case :
            if idpac[0] == 4:
                idpac = np.array([4, 0, 0])
                self._csuro = False
            if (idpac[1] == 0) or (idpac[2] == 0):
                self._csuro = False
                idpac = (idpac[0], 0, 0)
        self._idpac = idpac
        self.method, self.surro, self.norm = pacstr(idpac)

    def _speccheck(self, filt=None, dcomplex=None, filtorder=None, cycle=None,
                   width=None):
        """Check spectral parameters."""
        # Check the filter name :
        if filt is not None:
            if filt not in ['fir1', 'butter', 'bessel']:
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

    def _phampcheck(self, pha, amp, axis):
        """Check phase and amplitude values."""
        # Shape checking :
        if pha.ndim != amp.ndim:
            raise ValueError("pha and amp must have the same number of "
                             "dimensions.")
        # Force phase / amplitude to be at least (1, N) :
        if (pha.ndim == 1) and (amp.ndim == 1):
            pha = pha.reshape(1, -1)
            amp = amp.reshape(1, -1)
            axis = 1
        # Check if the phase is in radians :
        if np.ptp(pha) > 2 * np.pi:
            raise ValueError("Your phase is probably in degrees and should be"
                             " converted in radians using either np.degrees or"
                             " np.deg2rad.")
        # Check if the phase/amplitude have the same number of points on axis:
        if pha.shape[axis] != amp.shape[axis]:
            phan, ampn = pha.shape[axis], amp.shape[axis]
            raise ValueError("The phase (" + str(phan) + ") and the amplitude "
                             "(" + str(ampn) + ") do not have the same number"
                             " of points on the specified axis (" +
                             str(axis) + ").")
        # Force the phase to be in [-pi, pi] :
        pha = (pha + np.pi) % (2 * np.pi) - np.pi
        return pha, amp, axis

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
