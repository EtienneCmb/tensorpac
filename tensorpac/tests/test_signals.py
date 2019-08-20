"""Test random signals generation."""
import numpy as np

from tensorpac.signals import pac_signals_wavelet, pac_signals_tort


class TestSignals(object):
    """Test random signals generation."""

    def test_pac_signals_dtrials(self):
        """Definition of artificially coupled signals using dPha/dAmp."""
        assert pac_signals_tort(f_pha=5, f_amp=130, sf=512, n_epochs=23,
                                chi=0.9, noise=2, dpha=35, damp=46)

    def test_pac_signals_bandwidth(self):
        """Definition of artificially coupled signals using bandwidth."""
        assert pac_signals_tort(f_pha=[5, 7], f_amp=[30, 60], sf=200.,
                                n_epochs=100, chi=0.5, noise=3., n_times=1000)
        assert pac_signals_wavelet(f_pha=10, f_amp=57., n_times=1240, sf=256,
                                   noise=.7, n_epochs=33, pp=np.pi / 4,
                                   rnd_state=23)

    def test_default_args(self):
        """Test default aurguments for pac_vec."""
        assert pac_signals_tort(chi=2., noise=11., dpha=120., damp=200.)
