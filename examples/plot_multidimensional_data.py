"""
====================================
Compute PAC on multidimensional data
====================================

One of the strengths of Tensorpac is the ability to compute PAC on
multidimensional data. This example first generate a more realistic dataset
organized as (n_subjects, n_electrodes, n_trias, n_points) and compute PAC on
it.
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from tensorpac import Pac, pac_signals_wavelet

plt.style.use('seaborn-paper')

# ----------------- Create the dataset -----------------
# Dataset properties :
n_subject = 4  # Number of subjects
n_elec = 3     # Number of electrodes per subject
n_trials = 5   # Number of trials per subect and per electrode
n_pts = 2000   # Number of time points
sf = 256.      # Sampling frequency

# Generate different coupling for each subject/electrode :
f_low_pha, f_high_pha = 2., 10.    # lowest/highest centered frq. for phase
f_low_amp, f_high_amp = 30., 120.  # lowest/highest centered frq. for amplitude
low_noise, high_noise = .3, 1.5    # lowest/highest level of noise

st = 'S{s}, Elec{e}, Pha {p}hz, Amp {a}hz, Noise {n}'
data, info = np.zeros((n_subject, n_elec, n_trials, n_pts), dtype=float), []
for i in range(n_subject):
    for k in range(n_elec):
        # Generate a random centered phase/amplitude/noise level :
        fpha = np.random.randint(f_low_pha, f_high_pha, 1)[0]
        famp = np.random.randint(f_low_amp, f_high_amp, 1)[0]
        noise = np.random.uniform(low_noise, high_noise, 1)[0]
        # Get dataset informations : :
        inf = st.format(s=str(i + 1), e=str(k + 1), p=str(fpha), a=str(famp),
                        n=str(noise))
        info.append(inf)
        # Generate coupling :
        data[i, k, ...], time = pac_signals_wavelet(fpha=fpha, famp=famp,
                                                    noise=noise, npts=n_pts,
                                                    ntrials=n_trials, sf=sf)

# ----------------- Compute PAC -----------------
# Define the PAC object :
p = Pac(idpac=(4, 0, 0), fpha=(1, 15, 1, .3), famp=(25, 130, 5, 1),
        dcomplex='wavelet', width=12)
# Compute PAC along the time axis and take the mean across trials :
pac = p.filterfit(sf, data, axis=3).mean(-1)

# ----------------- Visualization -----------------
q = 1
plt.figure(figsize=(20, 19))
for i in range(n_subject):
    for k in range(n_elec):
        plt.subplot(n_subject, n_elec, q)
        p.comodulogram(pac[:, :, i, k], title=info[q - 1], interp=(.2, .2))
        q += 1

p.show()
