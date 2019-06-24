"""
==============
README example
==============

Reproduced the figure in the README.
"""
from tensorpac import Pac, pac_signals_tort

# Dataset of signals artificially coupled between 10hz and 100hz :
n_epochs = 20
n_times = 4000
sf = 512.  # sampling frequency

# Create artificially coupled signals using Tort method :
data, time = pac_signals_tort(f_pha=10, f_amp=100, noise=2, n_epochs=n_epochs,
                              dpha=10, damp=10, sf=sf, n_times=n_times)

# Define a PAC object :
p = Pac(idpac=(6, 0, 0), f_pha=(2, 20, 1, 1), f_amp=(60, 150, 5, 5))
# Filter the data and extract PAC :
xpac = p.filterfit(sf, data)

# Plot your Phase-Amplitude Coupling :
p.comodulogram(xpac.mean(-1), title='Contour plot with 5 regions',
               cmap='Spectral_r', plotas='contour', ncontours=5)

p.show()
