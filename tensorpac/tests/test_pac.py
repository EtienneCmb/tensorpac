"""Test tensorpac functions."""
import numpy as np
import matplotlib

from tensorpac import Pac, EventRelatedPac, PreferredPhase
from tensorpac.utils import pac_trivec


class TestPac(object):
    """Tests for Pac, Erpac and Preferred phase."""

    def test_id_pac_definition(self):
        """Test Pac object definition."""
        nmeth, nsuro, nnorm = 5, 4, 5
        for k in range(nmeth):
            for i in range(nsuro):
                for j in range(nnorm):
                    p = Pac(idpac=(k + 1, i, j))
                    str(p)

    def test_filtering_definition(self):
        """Test filtering defintion."""
        dcomplex = ['hilbert', 'wavelet']
        cycle = (12, 24)
        width = 12
        for k in dcomplex:
            Pac(dcomplex=k, cycle=cycle, width=width)

    def test_spectral(self):
        """Test filtering using the provided filters."""
        data = np.random.rand(3, 1000)
        p = Pac()
        dcomplex = ['hilbert', 'wavelet']
        for k in dcomplex:
            p.dcomplex = k
            p.filter(1024, data, n_jobs=1)

    def test_filter(self):
        """Test filter method."""
        data = np.random.rand(2, 1000)
        p = Pac()
        p.filter(256, data, 'phase')
        p.filter(256, data, 'amplitude')

    def test_fit(self):
        """Test all Pac methods."""
        pha = np.random.rand(2, 7, 1024)
        amp = np.random.rand(3, 7, 1024)
        nmeth, nsuro, nnorm = 5, 4, 5
        p = Pac()
        for k in range(nmeth):
            for i in range(nsuro):
                for j in range(nnorm):
                    p.idpac = (k + 1, i, j)
                    p.fit(pha, amp, n_jobs=1, n_perm=10)

    def test_filterfit(self):
        """Test filtering test computing PAC."""
        data = np.random.rand(2, 1024)
        p = Pac(idpac=(4, 0, 0))
        p.filterfit(1024, data, n_jobs=1)
        p.idpac = (1, 1, 1)
        p.filterfit(1024, data, data, n_jobs=1, n_perm=2)
        p.dcomplex = 'wavelet'
        p.filter(1024, data, n_jobs=1)

    def test_properties(self):
        """Test Pac properties."""
        p = Pac()
        # Idpac :
        p.idpac
        p.idpac = (2, 1, 1)
        # Dcomplex :
        p.dcomplex
        p.dcomplex = 'wavelet'
        # Cycle :
        p.cycle
        p.cycle = (12, 24)
        # Width :
        p.width
        p.width = 12

    def test_pac_comodulogram(self):
        """Test Pac object definition.

        This test works locally but failed on travis...
        """
        matplotlib.use('agg')
        f, tridx = pac_trivec()
        pac = np.random.rand(20, 10)
        pval = np.random.rand(20, 10)
        p = Pac(f_pha=np.arange(11), f_amp=np.arange(21))
        p.comodulogram(pac, rmaxis=True, dpaxis=True)
        p.comodulogram(pac, plotas='contour', pvalues=pval)
        p.comodulogram(pac, plotas='pcolor', pvalues=pval, levels=[.5, .7],
                       under='gray', over='red', bad='orange')
        p = Pac(f_pha=np.arange(11), f_amp=f)
        pac = np.random.rand(len(f))
        p.triplot(pac, f, tridx)
        matplotlib.pyplot.close('all')


class TestErpac(object):
    """Test EventRelatedPac class."""

    def test_filter(self):
        """Test function filter."""
        data = np.random.rand(7, 1000)
        p = EventRelatedPac()
        p.filter(256, data, 'phase')
        p.filter(256, data, 'amplitude')

    def test_fit(self):
        """Test function fit."""
        data = np.random.rand(100, 1000)
        p = EventRelatedPac()
        pha = p.filter(256, data, 'phase')
        amp = p.filter(256, data, 'amplitude')
        p.fit(pha, amp, method='circular')
        p.fit(pha, amp, method='gc')

    def test_filterfit(self):
        """Test function filterfit."""
        p = EventRelatedPac()
        x_pha = np.random.rand(100, 1000)
        x_amp = np.random.rand(100, 1000)
        p.filterfit(256, x_pha, x_amp=x_amp, method='circular')
        p.filterfit(256, x_pha, x_amp=x_amp, method='gc')


class TestPreferredPhase(object):
    """Test EventRelatedPac class."""

    def test_filter(self):
        """Test function filter."""
        data = np.random.rand(7, 1000)
        p = PreferredPhase()
        p.filter(256, data, 'phase')
        p.filter(256, data, 'amplitude')

    def test_fit(self):
        """Test function fit."""
        data = np.random.rand(100, 1000)
        p = PreferredPhase()
        pha = p.filter(256, data, 'phase')
        amp = p.filter(256, data, 'amplitude')
        p.fit(pha, amp)

    def test_filterfit(self):
        """Test function filterfit."""
        p = PreferredPhase()
        x_pha = np.random.rand(100, 1000)
        x_amp = np.random.rand(100, 1000)
        p.filterfit(256, x_pha, x_amp=x_amp)
