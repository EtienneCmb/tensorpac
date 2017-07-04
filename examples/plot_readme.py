"""
==============
README example
==============

Reproduced the figure in the README.
"""
from tensorpac.utils import pac_signals_tort
from tensorpac import Pac

# Dataset of signals artificially coupled between 10hz and 100hz :
n = 20     # number of datasets
sf = 512.  # sampling frequency

# Create artificially coupled signals using Tort method :
data, time = pac_signals_tort(fpha=10, famp=100, noise=2, ntrials=n,
                              dpha=10, damp=10, sf=sf)

# Define a PAC object :
p = Pac(idpac=(4, 0, 0), fpha=(2, 20, 1, 1), famp=(60, 150, 5, 5),
        dcomplex='wavelet', width=12)
# Filter the data and extract PAC :
xpac = p.filterfit(sf, data, axis=1)

# Plot your Phase-Amplitude Coupling :
p.comodulogram(xpac.mean(-1), title='Contour plot with 5 regions',
               cmap='Spectral_r', plotas='contour', ncontours=5)

p.show()
