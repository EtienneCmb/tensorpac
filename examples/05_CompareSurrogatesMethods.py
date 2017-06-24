"""This script compare the several surrogates evaluation methods."""
from __future__ import print_function
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

for i, k in enumerate(range(7)):
    # Change the pac method :
    p.idpac = (1, k, 3)
    print('-> Surrogates using '+p.surro)
    # Compute only the PAC without filtering :
    xpac, _ = p.fit(phases, amplitudes, axis=2, nperm=50)
    # Plot :
    plt.subplot(3, 3, k+1)
    p.comodulogram(xpac.mean(-1), title=p.surro, cmap='Spectral_r')

plt.show()
