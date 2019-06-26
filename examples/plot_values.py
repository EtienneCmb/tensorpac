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
# and 100hz. By default, this dataset is organized as (n_epochs, n_times) where
# n_times is the number of time points.
n_epochs = 1      # number of datasets
sf = 512.  # sampling frequency
data, time = pac_signals_wavelet(f_pha=6, f_amp=90, noise=.8,
                                 n_epochs=n_epochs, n_times=4000, sf=sf)


# First, let's use the MVL, without any further correction by surrogates :
p = Pac(idpac=(1, 2, 0), f_pha=(2, 15, 2, .2), f_amp=(60, 120, 10, 1))
xpac = p.filterfit(sf, data, n_perm=200, p=.05)
pval = p.pvalues

p.comodulogram(xpac.mean(-1), title=str(p), cmap='Spectral_r', vmin=0.,
               pvalues=pval, levels=.05)

p.show()
