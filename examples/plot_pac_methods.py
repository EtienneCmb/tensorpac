"""
===================
Compare PAC methods
===================

Compute PAC on multiple datasets and compare implemented methods.
"""
from __future__ import print_function
import matplotlib.pyplot as plt
from tensorpac import Pac, pac_signals_tort
plt.style.use('seaborn-paper')

# First, we generate a dataset of signals artificially coupled between 10hz
# and 100hz. By default, this dataset is organized as (n_trials, n_pts) where
# n_pts is the number of time points.
n = 10  # number of datasets
sf = 512.  # sampling frequency
n_pts = 4000  # Number of time points
data, time = pac_signals_tort(sf=sf, f_pha=5, f_amp=100, noise=2.,
                              n_trials=n, n_pts=n_pts)

# First, let's use the MVL, without any further correction by surrogates :
p = Pac(f_pha=(2, 20, 1, 1), f_amp=(60, 150, 5, 5))

# Now, we want to compare PAC methods, hence it's useless to systematically
# filter the data. So we extract the phase and the amplitude only once :
phases = p.filter(sf, data, ftype='phase')
amplitudes = p.filter(sf, data, ftype='amplitude')

plt.figure(figsize=(18, 9))
for i, k in enumerate([1, 2, 3, 4, 5, 6]):
    # Change the pac method :
    p.idpac = (k, 0, 0)
    print('-> PAC using ' + str(p))
    # Compute only the PAC without filtering :
    xpac = p.fit(phases, amplitudes).squeeze()
    # Plot :
    plt.subplot(2, 3, k)
    p.comodulogram(xpac.mean(-1), title=p.method, cmap='Spectral_r')

plt.show()
