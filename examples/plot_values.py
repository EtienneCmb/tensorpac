"""
================
Compute p-values
================

For the visualization, we used a comodulogram.
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorpac import Pac, pac_signals_wavelet
plt.style.use('seaborn-poster')

# First, we generate a dataset of signals artificially coupled between 10hz
# and 100hz. By default, this dataset is organized as (ntrials, npts) where
# npts is the number of time points.
n = 1      # number of datasets
sf = 512.  # sampling frequency
data, time = pac_signals_wavelet(fpha=6, famp=90, noise=.8, ntrials=n,
                                 npts=6000, sf=sf, rnd_state=3)


# First, let's use the MVL, without any further correction by surrogates :
p = Pac(idpac=(1, 2, 1), fpha=(2, 15, 2, .1), famp=(60, 120, 10, 1),
        dcomplex='wavelet', nblocks=10)
xpac, pval = p.filterfit(sf, data, axis=1, nperm=200, get_pval=True)
t1 = p.method + '\n' + p.surro + '\n' + p.norm

xpac, pval = np.squeeze(xpac), np.squeeze(pval)

plt.figure(figsize=(20, 7))
plt.subplot(121)
p.comodulogram(xpac, title=t1, cmap='Spectral_r', vmin=0.,
               bad='lightgray', pvalues=pval, p=0.01)

plt.subplot(122)
p.comodulogram(xpac, title=t1, cmap='Spectral_r', vmin=0.,
               pvalues=pval, levels=[0.01, 0.05])

p.show()
