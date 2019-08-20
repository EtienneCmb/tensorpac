"""Test gcmi related functions."""
import numpy as np

from tensorpac.gcmi import copnorm, nd_mi_gg
from tensorpac.config import CONFIG


class TestGcmi(object):
    """Test GCMI functions."""

    def test_copnorm(self):
        """Test Gaussian-copula normalization."""
        x = np.random.rand(10, 20, 30)
        copnorm(x)

    def test_mi(self):
        """Test computing MI."""
        rng = np.random.RandomState(0)
        x = rng.rand(100, 1, 100)
        y = rng.rand(100, 1, 100)
        # basic config
        nd_mi_gg(x, y)
        # modified config
        CONFIG['MI_DEMEAN'] = True
        CONFIG['MI_BIASCORRECT'] = True
        nd_mi_gg(x, y)
        CONFIG['MI_BIASCORRECT'] = False
        CONFIG['MI_DEMEAN'] = False
