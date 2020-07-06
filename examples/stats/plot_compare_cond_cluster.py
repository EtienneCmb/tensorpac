"""
========================================================================
Compare PAC of two experimental conditions with cluster-based statistics
========================================================================

This example illustrates how to statistically compare the phase-amplitude
coupling results coming from two experimental conditions. In particular, the
script below a the cluster-based approach to correct for the multiple
comparisons.

In order to work, this script requires MNE-Python package to be installed in
order to perform the cluster-based correction
(:func:`mne.stats.permutation_cluster_test`)
"""
import numpy as np

from tensorpac import Pac
from tensorpac.signals import pac_signals_wavelet

from mne.stats import permutation_cluster_test

import matplotlib.pyplot as plt


###############################################################################
# Simulate the data coming from two experimental conditions
###############################################################################
# Let's start by simulating data coming from two experimental conditions. The
# first dataset is going to simulate a 10hz phase <-> 120hz amplitude
# coupling while the second dataset will not include any coupling (random data)

# create the first dataset with a 10hz <-> 100hz coupling
n_epochs = 30   # number of datasets
sf = 512.       # sampling frequency
n_times = 4000  # Number of time points
x_1, time = pac_signals_wavelet(sf=sf, f_pha=10, f_amp=120, noise=2.,
                             n_epochs=n_epochs, n_times=n_times)
# create a second random dataset without any coupling
x_2 = np.random.rand(n_epochs, n_times)

###############################################################################
# Compute the single trial PAC on both datasets
###############################################################################
# once the datasets created, we can now extract the PAC, computed across
# time-points for each trials and across several phase and amplitude
# frequencies

# create the pac object. Use the Gaussian-Copula PAC
p = Pac(idpac=(6, 0, 0), f_pha='hres', f_amp='hres', dcomplex='wavelet')
# compute pac for both dataset
pac_1 = p.filterfit(sf, x_1, n_jobs=-1)
pac_2 = p.filterfit(sf, x_2, n_jobs=-1)

###############################################################################
# Correct for multiple-comparisons using a cluster-based approach
###############################################################################
# Then, we perform the cluster-based correction for multiple comparisons
# between the PAC coming from the two conditions. To this end we use the
# Python package MNE-Python and in particular, the function
# :func:`mne.stats.permutation_cluster_test`

# mne requires that the first is represented by the number of trials (n_epochs)
# Therefore, we transpose the output PACs of both conditions
pac_r1 = np.transpose(pac_1, (2, 0, 1))
pac_r2 = np.transpose(pac_2, (2, 0, 1))

n_perm = 1000  # number of permutations
tail = 1       # only inspect the upper tail of the distribution
# perform the correction
t_obs, clusters, cluster_p_values, h0 = permutation_cluster_test(
    [pac_r1, pac_r2], n_permutations=n_perm, tail=tail)

###############################################################################
# Plot the significant clusters
###############################################################################
# Finally, we plot the significant clusters. To this end, we used an elegant
# solution proposed by MNE where the non significant part appears using a
# gray scale colormap while significant clusters are going to be color coded.


# create new stats image with only significant clusters
t_obs_plot = np.nan * np.ones_like(t_obs)
for c, p_val in zip(clusters, cluster_p_values):
    if p_val <= 0.001:
        t_obs_plot[c] = t_obs[c]
        t_obs[c] = np.nan

title = 'Cluster-based corrected differences\nbetween cond 1 and 2'
p.comodulogram(t_obs, cmap='gray', colorbar=False)
p.comodulogram(t_obs_plot, cmap='viridis', title=title)
plt.gca().invert_yaxis()
plt.show()
