"""Test stastical functions."""
import numpy as np

from tensorpac.stats import test_stationarity as stats_stationarity

n_epochs = 8
n_times = 200
sf = 128.
pval = .05
rng = np.random.RandomState(1)
data = rng.rand(n_epochs, n_times)


class TestStats(object):
    """Test statistical functions."""

    def test_stats_stationarity(self):
        """Test the stationarity."""
        stats_stationarity(data, p=pval)
