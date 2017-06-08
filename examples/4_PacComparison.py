"""This script compare the several PAC methods.

Note that this script do not perform any correction by surrogates.
"""
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from tensorpac.utils import PacSignals
from tensorpac import Pac
plt.style.use('seaborn-paper')

# First, we generate a dataset of signals artificially coupled between 10hz
# and 100hz. By default, this dataset is organized as (ndatasets, npts) where
# npts is the number of time points.
n = 100  # number of datasets
sf = 1024  # sampling frequency
data, time = PacSignals(sf=sf, fpha=10, famp=100, noise=3, ndatasets=n,
                        dpha=10, damp=10)

# First, let's use the MVL, without any further correction by surrogates :
p = Pac(fpha=(1, 30, 1, 1), famp=(60, 160, 5, 5), dcomplex='wavelet', width=12)

# Now, we want to compare PAC methods, hence it's useless to systematically
# filter the data. So we extract the phase and the amplitude only once :
phases = p.filter(sf, data, axis=1, ftype='phase')
amplitudes = p.filter(sf, data, axis=1, ftype='amplitude')

titles = ['Mean Vector Length', 'Kullback-Leibler Divergence',
          'Heigh-Ratio', 'ndPAC']
for i, k in enumerate([1, 2, 3, 4]):
    # Change the pac method :
    p.idpac = (k, 0, 0)
    # Compute only the PAC without filtering :
    xpac, _ = p.fit(1024, phases, amplitudes, axis=2)
    # Plot :
    plt.subplot(3, 2, k)
    p.comodulogram(xpac.mean(-1), title=titles[i], cmap='Spectral_r')    

# The Phase-Synchrony needs the phase of the amplitude :
phaamplitudes = np.angle(hilbert(amplitudes))
p.idpac = (5, 0, 0)
xpac, _ = p.fit(1024, phases, phaamplitudes, axis=2)
plt.subplot(3, 2, 5)
p.comodulogram(xpac.mean(-1), title='Phase Synchrony', cmap='Spectral_r')

plt.show()
