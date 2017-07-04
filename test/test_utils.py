"""Test tensorpac utils."""
import numpy as np
from tensorpac.utils import (pac_vec, pac_signals_tort, pac_trivec,
                             pac_signals_wavelet)


def test_pac_vec():
    """Definition of PAC vectors."""
    assert pac_vec()
    assert pac_vec(fpha=(1, 30, 2, 2), famp=(60, 200, 10, 5))
    assert pac_vec(fpha=[1, 2], famp=np.arange(50))
    assert pac_vec(fpha=np.array([[2, 4], [5, 7], [9, 10]]),
                   famp=np.array([[30, 60], [60, 90], [100, 200]]).T)
    assert pac_vec(fpha=[[1, 2], [5, 7]], famp=[60, 150])


def test_pac_signals_dtrials():
    """Definition of artificially coupled signals using dPha/dAmp."""
    assert pac_signals_tort(fpha=5, famp=130, sf=512, ntrials=23, chi=0.9,
                            noise=2, dpha=35, damp=46)


def test_pac_signals_bandwidth():
    """Definition of artificially coupled signals using bandwidth."""
    assert pac_signals_tort(fpha=[5, 7], famp=[30, 60], sf=200., ntrials=100,
                            chi=0.5, noise=3., npts=1000)
    assert pac_signals_wavelet(fpha=10, famp=57., npts=1240, sf=256, noise=.7,
                               ntrials=33, pp=np.pi/4, rnd_state=23)


def test_default_args():
    """Test default aurguments for pac_vec."""
    assert pac_signals_tort(chi=2., noise=11., dpha=120., damp=200.)


def test_trivec():
    """Definition of triangular vectors."""
    assert pac_trivec(2, 200, 10)
