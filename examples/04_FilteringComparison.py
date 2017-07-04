"""
===============================
Filtering properties and effect
===============================

Tensorpac provides two ways for extracting phase and amplitude :

* Using filtering followed by Hilbert transform.
* Using wavelets.
"""
import matplotlib.pyplot as plt
from tensorpac.utils import pac_signals_tort
from tensorpac import Pac
plt.style.use('seaborn-paper')

# First, we generate a dataset of signals artificially coupled between 10hz
# and 100hz. By default, this dataset is organized as (ntrials, npts) where
# npts is the number of time points.
n = 10  # number of datasets
npts = 3000  # number of time points
data, time = pac_signals_tort(fpha=10, famp=100, noise=0, ntrials=n, npts=npts)

# First, let's use the MVL, without any further correction by surrogates :
p = Pac(idpac=(1, 0, 0), fpha=(5, 20, 2, 1), famp=(70, 130, 5, 5))

# Use several filter-order for the Butterworth filter :
p.filt = 'butter'
for i, k in enumerate([1, 3, 6]):
    p.filtorder = k
    xpac = p.filterfit(1024, data, axis=1)
    plt.subplot(3, 3, i + 1)
    p.comodulogram(xpac.mean(-1), title='Butterworth - order ' + str(k))

# Define several cycle options for the fir1 (eegfilt like) filter :
p.filt = 'fir1'
for i, k in enumerate([(3, 3), (3, 6), (6, 12)]):
    p.cycle = k
    xpac = p.filterfit(1024, data, axis=1)
    plt.subplot(3, 3, i + 4)
    p.comodulogram(xpac.mean(-1), title='Fir1 - cycle ' + str(k))

# Define several wavelet width :
p.dcomplex = 'wavelet'
for i, k in enumerate([7, 12, 24]):
    p.width = k
    xpac = p.filterfit(1024, data, axis=1)
    plt.subplot(3, 3, i + 7)
    p.comodulogram(xpac.mean(-1), title='Wavelet - width ' + str(k))

plt.show()
