"""
====================================================
Compararison of several surrogate evaluation methods
====================================================

Surrogates are used to generate a chance ditribution in order to correct the
PAC estimation.
"""
from __future__ import print_function
import matplotlib.pyplot as plt
from tensorpac.utils import pac_signals_wavelet
from tensorpac import Pac
plt.style.use('seaborn-paper')

# First, we generate a delta <-> low-gamma coupling. By default, this dataset
#  is organized as (ntrials, npts) where npts is the number of time points.
n = 30  # number of datasets
sf = 512.  # sampling frequency
data, time = pac_signals_wavelet(sf=sf, fpha=6, famp=70, noise=2., ntrials=n,
                                 npts=4000)

# First, let's use the MVL, without any further correction by surrogates :
p = Pac(fpha=(3, 10, 1, .2), famp=(50, 90, 5, 1), dcomplex='wavelet', width=12)

# Now, we want to compare PAC methods, hence it's useless to systematically
# filter the data. So we extract the phase and the amplitude only once :
phases = p.filter(sf, data, axis=1, ftype='phase')
amplitudes = p.filter(sf, data, axis=1, ftype='amplitude')

plt.figure(figsize=(18, 9))
for i, k in enumerate(range(5)):
    # Change the pac method :
    p.idpac = (5, k, 1)
    print('-> Surrogates using ' + p.surro)
    # Compute only the PAC without filtering :
    xpac = p.fit(phases, amplitudes, axis=2, nperm=5)
    # Plot :
    plt.subplot(2, 3, k + 1)
    p.comodulogram(xpac.mean(-1), title=p.surro, cmap='Spectral_r')

plt.show()
