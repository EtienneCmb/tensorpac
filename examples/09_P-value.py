"""
================
Compute p-values
================

For the visualization, we used a comodulogram.
"""
import matplotlib.pyplot as plt
from tensorpac.utils import pac_signals
from tensorpac import Pac
plt.style.use('seaborn-poster')

# First, we generate a dataset of signals artificially coupled between 10hz
# and 100hz. By default, this dataset is organized as (ndatasets, npts) where
# npts is the number of time points.
n = 1  # number of datasets
data, time = pac_signals(fpha=[8, 10], famp=[80, 100], noise=3, ndatasets=n,
                         chi=.8, npts=5000)

# First, let's use the MVL, without any further correction by surrogates :
p = Pac(idpac=(2, 2, 1), fpha=(2, 15, 1, .3), famp=(60, 120, 5, 1),
        dcomplex='wavelet')
xpac, pval = p.filterfit(1024, data, data, axis=1, nperm=100)
t1 = p.method + '\n' + p.surro + '\n' + p.norm

p.comodulogram(xpac[..., 0], title=t1, cmap='Spectral_r', vmin=0.,
               pvalues=pval[..., 0], bad='orange')

plt.show()
