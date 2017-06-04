"""Main PAC class."""
import numpy as np

from .utils import _CheckFreq
from .spectral import spectral
from .methods import ComputePac
from .surrogates import ComputeSurogates
from .normalize import normalize


class Pac(object):
    """Compute Phase-Amplitude Coupling using tensors."""

    def __init__(self, sf, idpac=(1, 1, 3), fpha=[2, 4], famp=[60, 200],
                 cycle=(3, 6), dcomplex='hilbert', filt='fir1', filtorder=3,
                 nbins=18):
        """Check and initialize."""
        # ----------------- CHECKING -----------------
        # Sampling frequency :
        if not isinstance(sf, (int, float)):
            raise ValueError("The sampling frequency must be a float number.")
        else:
            sf = float(sf)
        # Pac methods :
        idpac = np.atleast_1d(idpac)
        if not all([isinstance(k, int) for k in idpac]) and (len(idpac) != 3):
            raise ValueError("idpac must be a tuple/list of 3 integers.")
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
        self.sf = sf
        self.idpac = idpac
        self.fpha, self.famp = fpha, famp
        self.xvec, self.yvec = fpha.mean(1), famp.mean(1)
        self.filt, self.filtorder, self.cycle = filt, filtorder, cycle
        self.dcomplex = dcomplex
        self.nbins = nbins

    def __str__(self):
        """String representation."""
        pass

    def fit(self, xpha, xamp, axis=-1, traxis=-2, nperm=200, njobs=-1):
        p = 1/nperm
        # Extract phase (npha, ...) and amplitude (namp, ...) :
        pha = spectral(xpha, self.sf, self.fpha, axis, 'pha', self.dcomplex,
                       self.filt, self.filtorder, self.cycle[0], njobs)
        amp = spectral(xamp, self.sf, self.famp, axis, 'amp', self.dcomplex,
                       self.filt, self.filtorder, self.cycle[1], njobs)
        print('SHAPE : ', pha.shape, amp.shape)

        # Compute pac :
        pacargs = (self.idpac[0], self.nbins, p)
        pac = ComputePac(pha, amp, *pacargs)
        print('PAC : ', pac.shape)

        # Compute surogates :
        if self.idpac[1] != 0:
            surargs = (self.idpac[1], axis+1, traxis+1)
            suro = ComputeSurogates(pha, amp, surargs, pacargs, nperm, njobs)
            print('SURO : ', suro.shape)

            # Normalize pac by surrogates :
            pac = normalize(pac, np.mean(suro, axis=0, keepdims=0),
                            np.std(suro, axis=0, keepdims=0), self.idpac[2])

            # Compute statistics :

        return pac, suro.mean(0)
