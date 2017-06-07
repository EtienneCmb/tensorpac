"""Main PAC class."""
import numpy as np

from .utils import _CheckFreq
from .spectral import spectral
from .methods import ComputePac
from .surrogates import ComputeSurogates
from .normalize import normalize


class Pac(object):
    """Compute Phase-Amplitude Coupling (PAC) using tensors.

    Computing PAC is assessed in three steps : compute the real PAC, compute
    surrogates and finally, because PAC is very sensible to the noise, correct
    the real PAC by the surrogates. This implementation is modular i.e. it lets
    you choose among a large range of possible combinations.

    Kargs:
        idpac: tuple/list, optional, (def: (1, 1, 3))
            Choose the combination of methods to use in order to extract PAC.
            This tuple must be composed of three integers where each one them
            refer

            * First digit: refer to the pac method:

                - '1': Mean Vector Length (MVL) [#f1]_
                - '2': Kullback-Leibler Divergence (KLD) [#f2]_
                - '3': Heights Ratio (HR) [#f3]_
                - '4': ndPAC [#f4]_
                - '5': ERPAC

            * Second digit: refer to the method for computing surrogates:

                - '0': No surrogates
                - '1': Swap phase/amplitude across trials [#f2]_
                - '2': Swap amplitude time blocks [#f5]_
                - '3': Shuffle amplitude and phase time-series
                - '4': Shuffle phase time-series
                - '5': Shuffle amplitude time-series
                - '6': Time lag [#f1]_ [NOT IMPLEMENTED]
                - '7': Circular shifting [NOT IMPLEMENTED]

            * Third digit: refer to the normalization method for correction:

                - '0': No normalization
                - '1': Substract the mean of surrogates
                - '2': Divide by the mean of surrogates
                - '3': Substract then divide by the mean of surrogates
                - '4': Z-score

        fpha, famp: list/tuple/array, optional, (def: [2, 4] and [60, 200])
            Frequency vector for the phase and amplitude. Here you can use
            several forms to define those vectors :

                * Basic list/tuple (ex: [2, 4] or [8, 12]...)
                * List of frequency bands (ex: [[2, 4], [5, 7]]...)
                * Dynamic definition : (start, stop, width, step)
                * Range definition (ex : np.arange(3) => [[0, 1], [1, 2]])

        dcomplex: string, optional, (def: 'hilbert')
            Method for the complex definition. Use either 'hilbert' or
            'wavelet'.

        filt: string, optional, (def: 'fir1')
            Filtering method (only if dcomplex is 'hilbert'). Choose either
            'fir1', 'butter' or 'bessel'

        cycle: tuple, optional, (def: (3, 6))
            Control the number of cycles for filtering (only if dcomplex is
            'hilbert'). Should be a tuple of integers where the first one
            refers to the number of cycles for the phase and the second for the
            amplitude [#f5]_.

        filtorder: int, optional, (def: 3)
            Filter order for the Butterworth and Bessel filters (only if
            dcomplex is 'hilbert').

        nbins: int, optional, (def: 18)
            Number of bins for the KLD and HR PAC method [#f2]_ [#f3]_

    .. warning::
        * The ndPac [#f4]_ include a fast and reliable statistical test. As a
          result, if the ndPAC is choosed as the main PAC method, surrogates
          and normalization will be deactivate.

        * The phase in a particular frequency band can either be extracted
          using wavelet convolution or filtering followed by the Hilbert
          transform. As a result, every filtering related input (cycle, filt,
          filtorder) are going to be active if the complex decomposition is
          Hilbert.

    .. rubric:: Footnotes
    .. [#f1] `Canolty et al, 2006 <http://www.ncbi.nlm.nih.gov/pmc/articles/
       PMC2628289/>`_
    .. [#f2] `Tort et al, 2010 <http://www.ncbi.nlm.nih.gov/pmc/articles/
       PMC2941206/>`_
    .. [#f3] `Lakata et al, 2005 <https://www.ncbi.nlm.nih.gov/pubmed/
       15901760>`_
    .. [#f4] `Ozkurt et al, 2012 <http://www.ncbi.nlm.nih.gov/pubmed/
       22531738/>`_
    .. [#f5] `Bahramisharif et al, 2013 <http://www.jneurosci.org/content/33/
       48/18849.short/>`_
    """

    def __init__(self, idpac=(1, 1, 3), fpha=[2, 4], famp=[60, 200],
                 dcomplex='hilbert', filt='fir1', cycle=(3, 6), filtorder=3,
                 nbins=18):
        """Check and initialize."""
        self._csuro = True
        # ----------------- CHECKING -----------------
        # Pac methods :
        idpac = np.atleast_1d(idpac)
        if not all([isinstance(k, int) for k in idpac]) and (len(idpac) != 3):
            raise ValueError("idpac must be a tuple/list of 3 integers.")
        else:
            # Ozkurt PAC case :
            if idpac[0] == 4:
                idpac[1], idpac[2] = 0, 0
                self._csuro = False
            if (idpac[1] == 0) or (idpac[2] == 0):
                self._csuro = False
        # Frequency checking :
        fpha, famp = _CheckFreq(fpha), _CheckFreq(famp)
        # Check cycle :
        if (len(cycle) is not 2) or not all(isinstance(k, int) for k in cycle):
            raise ValueError("Cycle must be a tuple of two integers.")
        # Check complex decomposition :
        if dcomplex not in ['hilbert', 'wavelet']:
            raise ValueError("dcomplex must either be 'hilbert' or 'wavelet'.")
        # Check the filter name :
        if filt not in ['fir1', 'butter', 'bessel']:
            raise ValueError("filt must either be 'fir1', 'butter' or "
                             "'bessel'")
        # Convert filtorder and nbins :
        filtorder, nbins = int(filtorder), int(nbins)

        # ----------------- SELF -----------------
        self.idpac = idpac
        self.fpha, self.famp = fpha, famp
        self.xvec, self.yvec = fpha.mean(1), famp.mean(1)
        self.filt, self.filtorder, self.cycle = filt, filtorder, cycle
        self.dcomplex = dcomplex
        self.nbins = nbins

    def __str__(self):
        """String representation."""
        pass

    def fit(self, sf, xpha, xamp, axis=-1, traxis=-2, nperm=200, nblocks=2,
            correct=False, njobs=-1):
        """Run the defined PAC model on data.

        Args:
            sf: float
                The sampling frequency.

            xpha, xamp: np.ndarray
                Array of data for computing PAC. xpha is the data used for
                extracting phases and xamp, amplitudes. Both arrays must have
                the same shapes. If you want to compute PAC locally i.e. on the
                same electrode, x=xpha=xamp. For distant coupling, xpha and
                xamp could be different.

        Kargs:
            axis: int, optional, (def: -1)
                Dimension where is located the time axis. By default, the axis
                will be consider as well.

            traxis: int, optional, (def: -2)
                Dimension where is located the trial axis. By default the next-
                to-last axis is consider as the trial axis.

            nperm: int, optional, (def: 200)
                Number of surrogates to compute.

            nblocks: int, optional, (def: 2)
                Number of blocks for splitting the amplitude. Only active is
                the surrogate method is 2 [#f5]_.

            correct: bool, optional, (def: True)
                Correct the PAC estimation XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

            njobs: int, optional, (def: -1)
                Number of jobs to compute PAC in parallel.

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
        # ----------------- CHECKING -----------------
        # Sampling frequency :
        if not isinstance(sf, (int, float)):
            raise ValueError("The sampling frequency must be a float number.")
        else:
            sf = float(sf)
        if xpha.shape != xamp.shape:
            raise ValueError("The shape of xpha and xamp must be equals.")

        # Extract phase (npha, ...) and amplitude (namp, ...) :
        pha = spectral(xpha, sf, self.fpha, axis, 'pha', self.dcomplex,
                       self.filt, self.filtorder, self.cycle[0], njobs)
        amp = spectral(xamp, sf, self.famp, axis, 'amp', self.dcomplex,
                       self.filt, self.filtorder, self.cycle[1], njobs)
        print('SHAPE : ', pha.shape, amp.shape)

        # Compute pac :
        pacargs = (self.idpac[0], self.nbins, 1/nperm)
        pac = ComputePac(pha, amp, *pacargs)
        print('PAC : ', pac.shape)

        # Compute surogates :
        if self._csuro:
            surargs = (self.idpac[1], axis+1, traxis+1, nblocks)
            suro = ComputeSurogates(pha, amp, surargs, pacargs, nperm, njobs)
            print('SURO : ', suro.shape)

            # Normalize pac by surrogates :
            pac = normalize(pac, np.mean(suro, axis=0),
                            np.std(suro, axis=0), self.idpac[2])

            # Compute statistics :

        if correct:
            pac[pac < 0.] = 0.

        return pac, None
