"""
======================
Compare normalizations
======================

The normalization correspond on the method used to correct the PAC estimation
with the chance distribution.
"""
from __future__ import print_function
import matplotlib.pyplot as plt
from tensorpac import Pac
from tensorpac.signals import pac_signals_wavelet
plt.style.use('seaborn-paper')

# First, we generate a dataset of signals artificially coupled between 10hz
# and 100hz. By default, this dataset is organized as (n_epochs, n_times) where
# n_times is the number of time points.
n_epochs = 10  # number of datasets
sf = 512.  # sampling frequency
data, time = pac_signals_wavelet(sf=sf, f_pha=10, f_amp=100, noise=1.,
                                 n_epochs=n_epochs, n_times=2000)

# First, let's use the MVL, without any further correction by surrogates :
p = Pac(f_pha=(5, 16, 1, .1), f_amp=(80, 130, 5, 2))

# Now, we want to compare PAC methods, hence it's useless to systematically
# filter the data. So we extract the phase and the amplitude only once :
phases = p.filter(sf, data, ftype='phase', n_jobs=1)
amplitudes = p.filter(sf, data, ftype='amplitude', n_jobs=1)

plt.figure(figsize=(18, 9))
for i, k in enumerate(range(5)):
    # Change the pac method :
    p.idpac = (1, 2, k)
    print('-> Normalization using ' + p.str_norm)
    # Compute only the PAC without filtering :
    xpac = p.fit(phases, amplitudes, n_perm=20)
    # Plot :
    plt.subplot(2, 3, k + 1)
    p.comodulogram(xpac.mean(-1), title=p.str_norm, cmap='Spectral_r')

plt.show()
