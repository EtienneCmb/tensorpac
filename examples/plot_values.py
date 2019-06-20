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
# and 100hz. By default, this dataset is organized as (n_trials, n_pts) where
# n_pts is the number of time points.
n = 1      # number of datasets
sf = 512.  # sampling frequency
data, time = pac_signals_wavelet(f_pha=6, f_amp=90, noise=.8, n_trials=n,
                                 n_pts=6000, sf=sf, rnd_state=3)


# First, let's use the MVL, without any further correction by surrogates :
p = Pac(idpac=(1, 2, 1), f_pha=(2, 15, 2, .1), f_amp=(60, 120, 10, 1),
        dcomplex='wavelet')
xpac = p.filterfit(sf, data, n_perm=200).squeeze()
pval = p.pvalues_.squeeze()
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
