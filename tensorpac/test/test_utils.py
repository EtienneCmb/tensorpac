"""Test tensorpac utils."""
import numpy as np
from tensorpac.utils import (pac_vec, pac_signals_tort, pac_trivec,
                             pac_signals_wavelet)


def test_pac_vec():
    """Definition of PAC vectors."""
    assert pac_vec()
    assert pac_vec(f_pha=(1, 30, 2, 2), f_amp=(60, 200, 10, 5))
    assert pac_vec(f_pha=[1, 2], f_amp=np.arange(50))
    assert pac_vec(f_pha=np.array([[2, 4], [5, 7], [9, 10]]),
                   f_amp=np.array([[30, 60], [60, 90], [100, 200]]).T)
    assert pac_vec(f_pha=[[1, 2], [5, 7]], f_amp=[60, 150])


def test_pac_signals_dtrials():
    """Definition of artificially coupled signals using dPha/dAmp."""
    assert pac_signals_tort(f_pha=5, f_amp=130, sf=512, n_epochs=23, chi=0.9,
                            noise=2, dpha=35, damp=46)


def test_pac_signals_bandwidth():
    """Definition of artificially coupled signals using bandwidth."""
    assert pac_signals_tort(f_pha=[5, 7], f_amp=[30, 60], sf=200.,
                            n_epochs=100, chi=0.5, noise=3., n_times=1000)
    assert pac_signals_wavelet(f_pha=10, f_amp=57., n_times=1240, sf=256,
                               noise=.7, n_epochs=33, pp=np.pi/4, rnd_state=23)


def test_default_args():
    """Test default aurguments for pac_vec."""
    assert pac_signals_tort(chi=2., noise=11., dpha=120., damp=200.)


def test_trivec():
    """Definition of triangular vectors."""
    assert pac_trivec(2, 200, 10)
