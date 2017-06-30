"""This script compare the several surrogates evaluation methods."""
from __future__ import print_function
import matplotlib.pyplot as plt
from tensorpac.utils import pac_signals
from tensorpac import Pac
plt.style.use('seaborn-paper')

# First, we generate a delta <-> low-gamma coupling. By default, this dataset
#  is organized as (ndatasets, npts) where npts is the number of time points.
n = 100  # number of datasets
sf = 200.  # sampling frequency
npts = 4000  # number of time points
data, time = pac_signals(sf=sf, fpha=[5, 7], famp=[60, 80], noise=3,
                         ndatasets=n, npts=npts, chi=.9)

# First, let's use the MVL, without any further correction by surrogates :
p = Pac(fpha=(1, 10, 1, 1), famp=(50, 120, 5, 2), dcomplex='wavelet')

# Now, we want to compare PAC methods, hence it's useless to systematically
# filter the data. So we extract the phase and the amplitude only once :
phases = p.filter(sf, data, axis=1, ftype='phase')
amplitudes = p.filter(sf, data, axis=1, ftype='amplitude')

for i, k in enumerate(range(5)):
    # Change the pac method :
    p.idpac = (1, k, 3)
    print('-> Surrogates using ' + p.surro)
    # Compute only the PAC without filtering :
    xpac, _ = p.fit(phases, amplitudes, axis=2, nperm=20)
    # Plot :
    plt.subplot(3, 3, k + 1)
    p.comodulogram(xpac.mean(-1), title=p.surro, cmap='Spectral_r')

plt.show()
