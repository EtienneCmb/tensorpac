"""Test tensorpac utils."""
import numpy as np
import matplotlib

from tensorpac.utils import pac_vec, pac_trivec, PSD, BinAmplitude, ITC


class TestUtils(object):
    """Test utility functions."""

    def test_pac_vec(self):
        """Definition of PAC vectors."""
        assert pac_vec()
        assert pac_vec(f_pha=(1, 30, 2, 2), f_amp=(60, 200, 10, 5))
        assert pac_vec(f_pha=[1, 2], f_amp=np.arange(50))
        assert pac_vec(f_pha=np.array([[2, 4], [5, 7], [9, 10]]),
                       f_amp=np.array([[30, 60], [60, 90], [100, 200]]).T)
        assert pac_vec(f_pha=[[1, 2], [5, 7]], f_amp=[60, 150])
        assert pac_vec(f_pha='lres', f_amp='lres')
        assert pac_vec(f_pha='mres', f_amp='mres')
        assert pac_vec(f_pha='hres', f_amp='hres')

    def test_trivec(self):
        """Definition of triangular vectors."""
        assert pac_trivec(2, 200, 10)

    def test_psd(self):
        """Test PSD."""
        # test definition
        x = np.random.rand(10, 200)
        psd = PSD(x, 128)
        # test properties
        psd.freqs
        psd.psd
        # test plotting
        matplotlib.use('agg')
        psd.plot(confidence=None, log=True, grid=True, interp=.1)
        psd.plot(confidence=.95, log=False, grid=False, interp=None)
        psd.show()
        matplotlib.pyplot.close('all')

    def test_binned_amplitude(self):
        """Test binned amplitude."""
        # test definition
        x = np.random.rand(10, 200)
        binamp = BinAmplitude(x, 128)
        # test plot
        binamp.plot(unit='rad')
        binamp.plot(unit='deg')
        binamp.show()
        matplotlib.pyplot.close('all')
        # test properties
        binamp.phase
        binamp.amplitude

    def test_itc(self):
        """Test Phase-locking Value."""
        # test definition
        x = np.random.rand(10, 200)
        itc_1d = ITC(x, 128, f_pha=[2, 4])
        itc_2d = ITC(x, 128, f_pha=[[2, 4], [5, 7]])
        # test properties
        itc_1d.itc
        # test plot
        itc_1d.plot()
        itc_2d.plot()
        itc_1d.show()
        matplotlib.pyplot.close('all')
