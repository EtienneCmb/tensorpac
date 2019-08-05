"""
=================================================
Compute and plot the Power Spectrum Density (PSD)
=================================================

This example illustrate how to compute and plot the Power Spectrum Density
(PSD) of an electrophysiological dataset. The PSD should be used to check that
there is the presence of a clear peak, espacially for frequency phase.
"""
from tensorpac import pac_signals_tort
from tensorpac.utils import PSD

import matplotlib.pyplot as plt

# Dataset of signals artificially coupled between 10hz and 100hz :
n_epochs = 20
n_times = 4000
sf = 512.  # sampling frequency

# Create artificially coupled signals using Tort method :
data, time = pac_signals_tort(f_pha=10, f_amp=100, noise=4, n_epochs=n_epochs,
                              dpha=10, damp=10, sf=sf, n_times=n_times)

# compute the PSD
psd = PSD(data, sf)

# plot the mean PSD across trials between [5, 150]Hz with a 95th confidence
# interval
ax = psd.plot(confidence=95, f_min=5, f_max=150)

# finally display the figure
psd.show()
