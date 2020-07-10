"""Test tensorpac functions."""
import numpy as np
from scipy import stats

import matplotlib
import matplotlib.pyplot as plt

from tensorpac import Pac, EventRelatedPac, PreferredPhase
from tensorpac.utils import pac_trivec
from tensorpac.signals import pac_signals_wavelet



def normal(x, mu, sigma):
    return ( 2. * np.pi * sigma ** 2. ) ** -.5 * np.exp(
                -.5 * (x - mu) ** 2. / sigma ** 2. )


class TestPac(object):
    """Tests for Pac, Erpac and Preferred phase."""

    def test_id_pac_definition(self):
        """Test Pac object definition."""
        nmeth, nsuro, nnorm = 6, 4, 5
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
        p.filter(256, data, 'phase', edges=2)
        p.filter(256, data, 'amplitude')

    def test_fit(self):
        """Test all Pac methods."""
        pha = np.random.rand(2, 7, 1024)
        amp = np.random.rand(3, 7, 1024)
        nmeth, nsuro, nnorm = 5, 4, 5
        p = Pac(verbose=False)
        for k in range(nmeth):
            for i in range(nsuro):
                for j in range(nnorm):
                    p.idpac = (k + 1, i, j)
                    p.fit(pha, amp, n_jobs=1, n_perm=10)
                    if (i >= 1) and (k + 1 != 4):
                        for mcp in ['maxstat', 'fdr', 'bonferroni']:
                            p.infer_pvalues(mcp=mcp)

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
        p.comodulogram(np.random.rand(10, 10, 20))
        p.comodulogram(pac, rmaxis=True, dpaxis=True, interp=(.1, .1))
        p.comodulogram(pac, plotas='contour', pvalues=pval)
        p.comodulogram(pac, plotas='pcolor', pvalues=pval, levels=[.5, .7],
                       under='gray', over='red', bad='orange')
        p = Pac(f_pha=np.arange(11), f_amp=f)
        pac = np.random.rand(len(f))
        p.triplot(pac, f, tridx)
        p.savefig('test_savefig.png')
        p.show()
        matplotlib.pyplot.close('all')

    def test_functional_pac(self):
        """Test functionnal pac."""
        # generate a 10<->100hz ground truth coupling
        n_epochs, sf, n_times = 1, 512., 4000
        data, time = pac_signals_wavelet(f_pha=10, f_amp=100, noise=.8,
                                         n_epochs=n_epochs, n_times=n_times,
                                         sf=sf)
        # phase / amplitude extraction (single time)
        p = Pac(f_pha='lres', f_amp='lres', dcomplex='wavelet', width=12)
        phases = p.filter(sf, data, ftype='phase', n_jobs=1)
        amplitudes = p.filter(sf, data, ftype='amplitude', n_jobs=1)
        # ground truth array construction
        n_pha, n_amp = len(p.xvec), len(p.yvec)
        n_pix = int(n_pha * n_amp)
        gt = np.zeros((n_amp, n_pha), dtype=bool)
        b_pha = np.abs(p.xvec.reshape(-1, 1) - np.array([[9, 11]])).argmin(0)
        b_amp = np.abs(p.yvec.reshape(-1, 1) - np.array([[95, 105]])).argmin(0)
        gt[b_amp[0]:b_amp[1] + 1, b_pha[0]:b_pha[1] + 1] = True

        plt.figure(figsize=(12, 9))
        plt.subplot(2, 3, 1)
        p.comodulogram(gt, title='Gound truth', cmap='magma', colorbar=False)
        # loop over implemented methods
        for i, k in enumerate([1, 2, 3, 5, 6]):
            # compute only the pac
            p.idpac = (k, 2, 3)
            xpac = p.fit(phases, amplitudes, n_perm=200).squeeze()
            pval = p.pvalues.squeeze()
            is_coupling = pval <= .05
            # count the number of correct pixels. This includes both true
            # positives and true negatives
            acc = 100 * (is_coupling == gt).sum() / n_pix
            assert acc > 95.
            # build title of the figure (for sanity check)
            meth = p.method.replace(' (', '\n(')
            title = f"Method={meth}\nAccuracy={np.around(acc, 2)}%"
            # set to nan everywhere it's not significant
            xpac[~is_coupling] = np.nan
            vmin, vmax = np.nanmin(xpac), np.nanmax(xpac)
            # plot the results
            plt.subplot(2, 3, i + 2)
            p.comodulogram(xpac, colorbar=False, vmin=vmin, vmax=vmax,
                           title=title)
            plt.ylabel(''), plt.xlabel('')
        plt.tight_layout()
        plt.show()  # show on demand


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
        p.fit(pha, amp, method='gc', n_perm=2)
        p.fit(pha, amp, method='gc', smooth=5)
        p.surrogates, p.pvalues

    def test_filterfit(self):
        """Test function filterfit."""
        p = EventRelatedPac()
        x_pha = np.random.rand(100, 1000)
        x_amp = np.random.rand(100, 1000)
        p.filterfit(256, x_pha, x_amp=x_amp, method='circular')
        p.filterfit(256, x_pha, x_amp=x_amp, method='gc')

    def test_functional_erpac(self):
        """Test function test_functional_pac."""
        # erpac simultation
        n_epochs, n_times, sf, edges = 400, 1000, 512., 50
        x, times = pac_signals_wavelet(f_pha=10, f_amp=100, n_epochs=n_epochs,
                                       noise=.1, n_times=n_times, sf=sf)
        times = times[edges:-edges]
        # phase / amplitude extraction (single time)
        p = EventRelatedPac(f_pha=[8, 12], f_amp=(30, 200, 5, 5),
                            dcomplex='wavelet', width=12)
        kw = dict(n_jobs=1, edges=edges)
        phases = p.filter(sf, x, ftype='phase', **kw)
        amplitudes = p.filter(sf, x, ftype='amplitude', **kw)
        n_amp = len(p.yvec)
        # generate a normal distribution
        gt = np.zeros((n_amp, n_times - 2 * edges))
        b_amp = np.abs(p.yvec.reshape(-1, 1) - np.array([[80, 120]])).argmin(0)
        gt[b_amp[0]:b_amp[1] + 1, :] = True

        plt.figure(figsize=(16, 5))
        plt.subplot(131)
        p.pacplot(gt, times, p.yvec, title='Ground truth', cmap='magma')

        for n_meth, meth in enumerate(['circular', 'gc']):
            # compute erpac + p-values
            erpac = p.fit(phases, amplitudes, method=meth,
                          mcp='bonferroni', n_perm=30).squeeze()
            pvalues = p.pvalues.squeeze()
            # find everywhere erpac is significant + compare to ground truth
            is_signi = pvalues < .05
            erpac[~is_signi] = np.nan
            # computes accuracy
            acc = 100 * (is_signi == gt).sum() / (n_amp * n_times)
            assert acc > 80.
            # plot the result
            title = f"Method={p.method}\nAccuracy={np.around(acc, 2)}%"
            plt.subplot(1, 3, n_meth + 2)
            p.pacplot(erpac, times, p.yvec, title=title)
        plt.tight_layout()
        plt.show()



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

    def test_polar_plot(self):
        """Test the polar plot."""
        matplotlib.use('agg')
        p = PreferredPhase(f_pha=[5, 7], f_amp=(60, 200, 10, 1))
        x_pha = np.random.rand(100, 1000)
        x_amp = np.random.rand(100, 1000)
        ampbin, pp, vecbin = p.filterfit(256, x_pha, x_amp=x_amp)
        pp = np.squeeze(pp).T
        ampbin = np.squeeze(ampbin).mean(-1)
        p.polar(ampbin.T, vecbin, p.yvec, cmap='RdBu_r', interp=.1,
                cblabel='Amplitude bins')
        matplotlib.pyplot.close('all')


if __name__ == '__main__':
    TestPac().test_fit()
    # TestErpac().test_functional_erpac()