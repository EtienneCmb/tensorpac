"""Test PAC methods."""
import numpy as np
from tensorpac.methods.meth_switch import get_pac_fcn, pacstr
from tensorpac.methods.meth_surrogates import compute_surrogates, normalize
from tensorpac.methods.meth_pp import preferred_phase
from tensorpac.methods.meth_erpac import erpac, ergcpac, _ergcpac_perm

n_pac_range = range(1, 7)
n_sur_range = range(4)
n_norm_range = range(1, 5)

n_epochs = 5
n_times = 1000
n_pha_freqs = 2
n_amp_freqs = 3
n_bins = 18
pval = .05
n_perm = 2

pha = np.random.uniform(-np.pi, np.pi, (n_pha_freqs, n_epochs, n_times))
amp = np.random.rand(n_amp_freqs, n_epochs, n_times)


class TestMethods(object):
    """Test individual pac methods."""

    @staticmethod
    def _get_methods(implementation='tensor'):
        meths = []
        q = 1
        while q is not None:
            try:
                meths += [get_pac_fcn(q, n_bins, pval, implementation)]
                q += 1
            except KeyError as e:  # noqa
                q = None
        return meths

    def test_pacstr(self):
        """Test getting pac string name."""
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
        for imp in ['tensor']:  # 'numba' = FAIL (core dumped)
            for n, meth in enumerate(self._get_methods(imp)):
                # print(meth.func.__name__)
                if n + 1 == 6:  # gc pac need additional multivariate axis
                    _pha = np.stack([np.sin(pha), np.cos(pha)], axis=-2)
                    _amp = amp[..., np.newaxis, :]
                    pac = meth(_pha, _amp)
                elif n + 1 == 4:  # Try with different values of p for coverage
                    pac = meth(pha, amp, p=0.5)
                    pac = meth(pha, amp, p=1)
                    pac = meth(pha, amp, p=None)
                else:
                    pac = meth(pha, amp)
                assert pac.shape == (n_amp_freqs, n_pha_freqs, n_epochs)

    def test_surrogates(self):
        """Test computing surrogates."""
        fcn = get_pac_fcn(1, n_bins, pval)
        s_shape = (n_perm, n_amp_freqs, n_pha_freqs, n_epochs)
        for s in n_sur_range:
            surro = compute_surrogates(pha, amp, s, fcn, n_perm, 1, 0)
            assert (surro is None) or (surro.shape == s_shape)

    def test_normalize(self):
        """Test normalization."""
        for k in n_norm_range:
            true_pac = np.random.rand(n_amp_freqs, n_pha_freqs)
            perm_pac = np.random.rand(n_perm, n_amp_freqs, n_pha_freqs)
            normalize(k, true_pac, perm_pac)

    def test_erpac(self):
        """Test event-related PAC."""
        er_pha, er_amp = np.moveaxis(pha, -2, -1), np.moveaxis(amp, -2, -1)
        # circular
        er_circ, pv_circ = erpac(er_pha, er_amp)
        assert er_circ.shape == pv_circ.shape
        assert er_circ.shape == (n_amp_freqs, n_pha_freqs, n_times)
        # gaussian copula
        _pha = np.stack([np.sin(er_pha), np.cos(er_pha)], axis=-2)
        _amp = er_amp[..., np.newaxis, :]
        ergc_circ = ergcpac(_pha, _amp, smooth=None)
        assert ergc_circ.shape == (n_amp_freqs, n_pha_freqs, n_times)
        ergcpac(_pha, _amp, smooth=5)
        # test erpac permutations
        ergc_perm = _ergcpac_perm(_pha, _amp, smooth=None, n_perm=n_perm)
        assert ergc_perm.shape == (n_perm, n_amp_freqs, n_pha_freqs, n_times)

    def test_preferred_phase(self):
        """Test preferred phase method."""
        bin_amp, pp, fvec = preferred_phase(pha, amp, n_bins)
        assert bin_amp.shape == (n_bins, n_amp_freqs, n_pha_freqs, n_epochs)
        assert pp.shape == (n_amp_freqs, n_pha_freqs, n_epochs)
        assert fvec.shape == (n_bins,)
