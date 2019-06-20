"""Test tensorpac functions."""
import numpy as np
import matplotlib
from tensorpac import Pac, pac_trivec


def test_id_pac_definition():
    """Test Pac object definition."""
    nmeth, nsuro, nnorm = 5, 4, 5
    for k in range(nmeth):
        for i in range(nsuro):
            for j in range(nnorm):
                p = Pac(idpac=(k + 1, i, j))
                str(p)


def test_filtering_definition():
    """Test filtering defintion."""
    dcomplex = ['hilbert', 'wavelet']
    filt = ['butter', 'fir1', 'bessel']
    cycle = (12, 24)
    filtorder = 4
    width = 12
    for k in dcomplex:
        for i in filt:
            Pac(dcomplex=k, filt=i, filtorder=filtorder, cycle=cycle,
                width=width)


def test_spectral():
    """Test filtering using the provided filters."""
    data = np.random.rand(3, 1, 1000)
    p = Pac()
    dcomplex = ['hilbert', 'wavelet']
    filt = ['butter', 'fir1', 'bessel']
    for k in dcomplex:
        for i in filt:
            p.dcomplex = k
            p.filt = i
            p.filter(1024, data, n_jobs=1)


def test_pac_meth():
    """Test all Pac methods."""
    pha = np.random.rand(2, 7, 2, 1024)
    amp = np.random.rand(3, 7, 2, 1024)
    nmeth, nsuro, nnorm = 5, 4, 5
    p = Pac()
    for k in range(nmeth):
        for i in range(nsuro):
            for j in range(nnorm):
                p.idpac = (k + 1, i, j)
                p.fit(pha, amp, n_jobs=1, n_perm=10)


def test_compute():
    """Test filtering test computing PAC."""
    data = np.random.rand(2, 1, 1024)
    p = Pac(idpac=(4, 0, 0))
    p.filterfit(1024, data, n_jobs=1)
    p.idpac = (1, 1, 1)
    p.filterfit(1024, data, data, n_jobs=1, n_perm=2)
    p.dcomplex = 'wavelet'
    p.filter(1024, data, n_jobs=1)


def test_properties():
    """Test Pac properties."""
    p = Pac()
    # Idpac :
    p.idpac
    p.idpac = (2, 1, 1)
    # Filt :
    p.filt
    p.filt = 'butter'
    # Dcomplex :
    p.dcomplex
    p.dcomplex = 'wavelet'
    # Cycle :
    p.cycle
    p.cycle = (12, 24)
    # Filtorder :
    p.filtorder
    p.filtorder = 6
    # Width :
    p.width
    p.width = 12


def test_pac_comodulogram():
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
    p.polar(pac, np.arange(10), np.arange(20), interp=.8)
    pac = np.random.rand(len(f))
    p.triplot(pac, f, tridx)
    matplotlib.pyplot.close('all')


def test_preferred_phase():
    """Test the prefered phase."""
    data = np.random.rand(2, 1, 1024)
    p = Pac(idpac=(4, 0, 0))
    pha = p.filter(1024, data, ftype='phase')
    amp = p.filter(1024, data, ftype='amplitude')
    p.pp(pha, amp)


def test_erpac():
    """Test the ERPAC."""
    data = 10 * np.random.rand(30, 1, 2024)
    p = Pac(idpac=(4, 0, 0))
    pha = p.filter(512, data, ftype='phase')
    amp = p.filter(512, data, ftype='amplitude')
    p.erpac(pha, amp, method='gc')
    p.erpac(pha, amp, method='circular')
