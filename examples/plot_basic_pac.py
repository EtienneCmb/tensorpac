"""
==============================
Basic phase amplitude coupling
==============================

For the visualization, we used a comodulogram.
"""
import matplotlib.pyplot as plt
from tensorpac.utils import pac_signals_tort
from tensorpac import Pac
plt.style.use('seaborn-poster')

# First, we generate a dataset of signals artificially coupled between 10hz
# and 100hz. By default, this dataset is organized as (n_trials, n_pts) where
# n_pts is the number of time points.
n = 10  # number of datasets
n_pts = 4000  # number of time points
sf = 512.
data, time = pac_signals_tort(f_pha=10, f_amp=100, noise=3., n_trials=n,
                              n_pts=n_pts, dpha=10, damp=5, chi=.8, sf=sf)

# First, let's use the MVL, without any further correction by surrogates :
p = Pac(idpac=(1, 0, 0), f_pha=(2, 30, 1, .5), f_amp=(60, 150, 10, 1))

# extract phases and amplitudes
phases = p.filter(sf, data, ftype='phase')
amplitudes = p.filter(sf, data, ftype='amplitude')

# compute pac without permutations
xpac = p.fit(phases, amplitudes).squeeze()
t1 = p.method + '\n' + p.surro + '\n' + p.norm

# Now, we still use the MVL method, but in addition we shuffle amplitude time
# series and then, subtract then divide by the mean of surrogates :
p.idpac = (1, 1, 3)
xpac_corr = p.fit(phases, amplitudes, n_perm=20).squeeze()
t2 = p.method + '\n' + p.surro + '\n' + p.norm

# plot the result by taking the mean across trials
plt.figure(figsize=(20, 7))
plt.subplot(1, 2, 1)
p.comodulogram(xpac.mean(-1), title=t1)

plt.subplot(1, 2, 2)
p.comodulogram(xpac_corr.mean(-1), title=t2)
plt.show()
