"""
============================
Compare filtering properties
============================

Tensorpac provides two ways for extracting phase and amplitude :

* Using filtering followed by Hilbert transform.
* Using wavelets.
"""
from __future__ import print_function
import matplotlib.pyplot as plt
from tensorpac.utils import pac_signals_wavelet
from tensorpac import Pac
plt.style.use('seaborn-paper')

# First, we generate a dataset of signals artificially coupled between 10hz
# and 100hz. By default, this dataset is organized as (ntrials, npts) where
# npts is the number of time points.
n = 5  # number of datasets
npts = 2000  # number of time points
data, time = pac_signals_wavelet(fpha=10, famp=100, noise=1., ntrials=n,
                                 npts=npts)

# First, let's use the MVL, without any further correction by surrogates :
p = Pac(idpac=(1, 0, 0), fpha=(5, 14, 2, .3), famp=(80, 120, 2, 1))

plt.figure(figsize=(18, 9))
# Define several cycle options for the fir1 (eegfilt like) filter :
p.filt = 'fir1'
print('Filtering with fir1 filter')
for i, k in enumerate([(1, 3), (3, 6), (6, 12)]):
    p.cycle = k
    xpac = p.filterfit(1024, data, axis=1)
    plt.subplot(2, 3, i + 1)
    p.comodulogram(xpac.mean(-1), title='Fir1 - cycle ' + str(k))

# Define several wavelet width :
p.dcomplex = 'wavelet'
print('Filtering with wavelets')
for i, k in enumerate([7, 12, 24]):
    p.width = k
    xpac = p.filterfit(1024, data, axis=1)
    plt.subplot(2, 3, i + 4)
    p.comodulogram(xpac.mean(-1), title='Wavelet - width ' + str(k))

plt.show()
