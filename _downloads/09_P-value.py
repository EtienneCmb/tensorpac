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
n = 1      # number of datasets
sf = 256.  # sampling frequency
data, time = pac_signals(fpha=6, famp=90, noise=2, ndatasets=n,
                         chi=.5, npts=5000, sf=sf)

# First, let's use the MVL, without any further correction by surrogates :
p = Pac(idpac=(1, 2, 1), fpha=(2, 15, 2, .1), famp=(60, 120, 10, 1),
        dcomplex='wavelet')
xpac, pval = p.filterfit(sf, data, data, axis=1, nperm=100)
t1 = p.method + '\n' + p.surro + '\n' + p.norm

p.comodulogram(xpac[..., 0], title=t1, cmap='Spectral_r', vmin=0.,
               bad='orange', pvalues=pval[..., 0], p=0.05)

plt.show()
