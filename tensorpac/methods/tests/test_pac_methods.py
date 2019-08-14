"""Test PAC methods."""
import numpy as np
from tensorpac.methods.meth_pac import get_pac_fcn, pacstr
from tensorpac.methods.meth_pp import preferred_phase

n_pac_range = range(1, 7)
n_sur_range = range(4)
n_norm_range = range(5)

n_epochs = 5
n_times = 1000
n_pha_freqs = 2
n_amp_freqs = 3
n_bins = 18
pval = .05

pha = np.random.uniform(-np.pi, np.pi, (n_pha_freqs, n_epochs, n_times))
amp = np.random.rand(n_amp_freqs, n_epochs, n_times)


class TestMethods(object):
    """Test individual pac methods."""

    @staticmethod
    def _get_methods():
        meths = []
        q = 1
        while q is not None:
            try:
                meths += [get_pac_fcn(q, n_bins, pval)]
                q += 1
            except ValueError as e:  # noqa
                q = None
        return meths

    def test_pacstr(self):
        """Test getting pas string name."""
        for p in n_pac_range:
            for s in n_sur_range:
                for n in n_norm_range:
                    names = pacstr((p, s, n))
                    assert all([isinstance(k, str) for k in names])
        # test assert error
        try:
            pacstr((p + 1, s, n))
        except ValueError as e:  # noqa
            pass
        try:
            pacstr((p, s + 1, n))
        except ValueError as e:  # noqa
            pass
        try:
            pacstr((p, s, n + 1))
        except ValueError as e:  # noqa
            pass

    def test_pac_methods(self):
        """Test individual pac methods."""
        for n, meth in enumerate(self._get_methods()):
            if n + 1 == 6:  # gc pac need additional multivariate axis
                _pha = np.stack([np.sin(pha), np.cos(pha)], axis=-2)
                _amp = amp[..., np.newaxis, :]
                pac = meth(_pha, _amp)
            else:
                pac = meth(pha, amp)
            assert pac.shape == (n_amp_freqs, n_pha_freqs, n_epochs)

    def test_preferred_phase(self):
        """Test preferred phase method."""
        bin_amp, pp, fvec = preferred_phase(pha, amp, n_bins)
        assert bin_amp.shape == (n_bins, n_amp_freqs, n_pha_freqs, n_epochs)
        assert pp.shape == (n_amp_freqs, n_pha_freqs, n_epochs)
        assert fvec.shape == (n_bins,)
