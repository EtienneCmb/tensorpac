"""
====================================
Compute PAC on multidimensional data
====================================

One of the strengths of Tensorpac is the ability to compute PAC on
multidimensional data. This example first generate a more realistic dataset
organized as (n_trials, n_channels, n_points) and compute PAC on it.
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from tensorpac import Pac, pac_signals_wavelet

plt.style.use('seaborn-paper')

# ----------------- Create the dataset -----------------
# Dataset properties :
n_channels = 3     # Number of electrodes
n_trials = 5       # Number of trials per subect and per electrode
n_pts = 2000       # Number of time points
sf = 256.          # Sampling frequency

# Generate different coupling for each channel :
f_low_pha, f_high_pha = 2., 10.    # lowest/highest centered frq. for phase
f_low_amp, f_high_amp = 30., 120.  # lowest/highest centered frq. for amplitude
low_noise, high_noise = .3, 1.5    # lowest/highest level of noise

# random state for reproducibility
rnd = np.random.RandomState(0)

st = 'Chan%s, Pha %shz, Amp %shz, Noise %s'
data, info = np.zeros((n_trials, n_channels, n_pts), dtype=float), []
for k in range(n_channels):
    # Generate a random centered phase/amplitude/noise level :
    f_pha = rnd.randint(f_low_pha, f_high_pha, 1)[0]
    f_amp = rnd.randint(f_low_amp, f_high_amp, 1)[0]
    noise = rnd.uniform(low_noise, high_noise, 1)[0]
    # Get dataset informations : :
    info += [st % (str(k + 1), str(f_pha), str(f_amp), str(noise))]
    # Generate coupling :
    data[:, [k], :], time = pac_signals_wavelet(f_pha=f_pha, f_amp=f_amp,
                                                noise=noise, n_pts=n_pts,
                                                n_trials=n_trials, sf=sf)

# ----------------- Compute PAC -----------------
# Define the PAC object :
p = Pac(idpac=(4, 0, 0), f_pha=(1, 15, 1, .3), f_amp=(25, 130, 5, 1),
        dcomplex='wavelet', width=12)
# Compute PAC along the time axis and take the mean across trials :
pac = p.filterfit(sf, data).mean(2)

# ----------------- Visualization -----------------
q = 1
plt.figure(figsize=(20, 19))
for k in range(n_channels):
    plt.subplot(1, n_channels, q)
    p.comodulogram(pac[:, :, k], title=info[q - 1], interp=(.2, .2))
    q += 1

p.show()
