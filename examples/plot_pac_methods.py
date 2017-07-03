"""
======================
PAC methods comparison
======================

Compute PAC on multiple datasets and compare implemented methods.
"""
from __future__ import print_function
import matplotlib.pyplot as plt
from tensorpac.utils import pac_signals
from tensorpac import Pac
plt.style.use('seaborn-paper')

# First, we generate a dataset of signals artificially coupled between 10hz
# and 100hz. By default, this dataset is organized as (ndatasets, npts) where
# npts is the number of time points.
n = 10  # number of datasets
sf = 256.  # sampling frequency
npts = 5000  # Number of time points
data, time = pac_signals(sf=sf, fpha=[4, 6], famp=[90, 110], noise=3,
                         ndatasets=n, dpha=10, damp=10, npts=npts)

# First, let's use the MVL, without any further correction by surrogates :
p = Pac(fpha=(1, 10, 1, .1), famp=(60, 140, 1, 1), dcomplex='wavelet',
        width=6)

# Now, we want to compare PAC methods, hence it's useless to systematically
# filter the data. So we extract the phase and the amplitude only once :
phases = p.filter(sf, data, axis=1, ftype='phase')
amplitudes = p.filter(sf, data, axis=1, ftype='amplitude')

plt.figure(figsize=(18, 9))
for i, k in enumerate([1, 2, 3, 4, 5]):
    # Change the pac method :
    p.idpac = (k, 0, 0)
    print('-> PAC using ' + str(p))
    # Compute only the PAC without filtering :
    xpac, _ = p.fit(phases, amplitudes, axis=2)
    # Plot :
    plt.subplot(2, 3, k)
    p.comodulogram(xpac.mean(-1), title=p.method, cmap='Spectral_r')

plt.show()
