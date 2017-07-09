"""Test tensorpac functions."""
import numpy as np
import matplotlib
from tensorpac import Pac
from tensorpac.utils import pac_trivec


def test_id_pac_definition():
    """Test Pac object definition."""
    nmeth, nsuro, nnorm = 5, 5, 5
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
    data = np.random.rand(3, 1000)
    p = Pac()
    dcomplex = ['hilbert', 'wavelet']
    filt = ['butter', 'fir1', 'bessel']
    for k in dcomplex:
        for i in filt:
            p.dcomplex = k
            p.filt = i
            p.filter(1024, data, axis=1, njobs=1)


def test_pac_meth():
    """Test all Pac methods."""
    pha = np.random.rand(2, 7, 1024)
    amp = np.random.rand(3, 7, 1024)
    nmeth, nsuro, nnorm = 5, 5, 5
    p = Pac()
    for k in range(nmeth):
        for i in range(nsuro):
            for j in range(nnorm):
                p.idpac = (k + 1, i, j)
                p.fit(pha, amp, axis=2, traxis=1, njobs=1, nperm=2)


def test_compute():
    """Test filtering test computing PAC."""
    data = np.random.rand(2, 1024)
    p = Pac(idpac=(4, 0, 0))
    p.filterfit(1024, data, njobs=1)
    p.idpac = (1, 1, 1)
    p.filterfit(1024, data, data, njobs=1, nperm=2)
    p.dcomplex = 'wavelet'
    p.filter(1024, data, axis=1, njobs=1)


def test_kargs():
    """Test filtering test computing PAC."""
    data = np.random.rand(2, 1024)
    p = Pac(idpac=(1, 2, 4))
    karg1 = p.filterfit(1024, data, data, njobs=1)
    karg2 = p.filterfit(1024, data, data, njobs=1, get_pval=True)
    karg3 = p.filterfit(1024, data, data, njobs=1, get_surro=True)
    karg4 = p.filterfit(1024, data, data, njobs=1, get_surro=True,
                        get_pval=True)
    assert len(karg1) == 1
    assert len(karg2) == 2
    assert len(karg3) == 2
    assert len(karg4) == 3


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
    p = Pac(fpha=np.arange(11), famp=np.arange(21))
    print(len(p.xvec), len(p.yvec))
    p.comodulogram(pac, rmaxis=True, dpaxis=True)
    p.comodulogram(pac, plotas='contour', pvalues=pval)
    p.comodulogram(pac, plotas='pcolor', pvalues=pval, levels=[.5, .7],
                   under='gray', over='red', bad='orange')
    p = Pac(fpha=np.arange(11), famp=f)
    p.polar(pac, np.arange(10), np.arange(20), interp=.8)
    pac = np.random.rand(len(f))
    p.triplot(pac, f, tridx)
    matplotlib.pyplot.close('all')


def test_preferred_phase():
    """Test the prefered phase."""
    data = np.random.rand(2, 1024)
    p = Pac(idpac=(4, 0, 0))
    pha = p.filter(1024, data, axis=1, ftype='phase')
    amp = p.filter(1024, data, axis=1, ftype='amplitude')
    p.pp(pha, amp, axis=2)


def test_erpac():
    """Test the ERPAC."""
    data = np.random.rand(2, 1024)
    p = Pac(idpac=(4, 0, 0))
    pha = p.filter(1024, data, axis=1, ftype='phase')
    amp = p.filter(1024, data, axis=1, ftype='amplitude')
    p.erpac(pha, amp, traxis=1)
