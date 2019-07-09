"""
======================================
Event-Related Phase Amplitude Coupling
======================================

Event-Related Phase-Amplitude Coupling (ERPAC) do not measure PAC across time
cycle but instead, across trials (just as proposed JP. Lachaux with the
PLV/PLS). Measuring across trials enable to have a real-time estimation of PAC.

In this example, we generate a signal that have a 10hz phase <->100 hz
amplitude coupling first followed by a random noise.
"""
import numpy as np
from tensorpac import EventRelatedPac, pac_signals_wavelet
import matplotlib.pyplot as plt

###############################################################################
# Generate a synthetic signal
###############################################################################
# in order to illustrate how the ERPAC does works, we are going to concatenate
# two signals. A first one with an alpha <-> gamma coupling during one second
# and then a second one which is going to be a one second random noise

# First signal consisting of a one second 10 <-> 100hz coupling
n_epochs = 300
n_times = 1000
sf = 1000.
x1, tvec = pac_signals_wavelet(f_pha=10, f_amp=100, n_epochs=n_epochs, noise=2,
                               n_times=n_times, sf=sf)

# Second signal : one second of random noise
x2 = np.random.rand(n_epochs, 1000)

# now, concatenate the two signals across the time axis
x = np.concatenate((x1, x2), axis=1)
time = np.arange(x.shape[1]) / sf


###############################################################################
# Define an ERPAC object and extract the phase and the amplitude
###############################################################################
# use :class:`tensorpac.EventRelatedPac.filter` method to extract phases and
# amplitudes

# define an ERPAC object
p = EventRelatedPac(f_pha=[9, 11], f_amp=(60, 140, 10, 10))

# extract phases and amplitudes
pha = p.filter(sf, x, ftype='phase')
amp = p.filter(sf, x, ftype='amplitude')

###############################################################################
# Compute the ERPAC using the two implemented methods and plot it
###############################################################################

# implemented ERPAC methods
# methods = ['circular', 'gc']

# plt.figure(figsize=(16, 8))
# for n_m, m in enumerate(methods):
#     # compute the erpac
#     erpac = p.fit(pha, amp, method=m, smooth=100).squeeze()

#     # plot
#     plt.subplot(len(methods), 1, n_m + 1)
#     p.pacplot(erpac, time, p.yvec, xlabel='Time (second)' * n_m,
#               cmap='Spectral_r', ylabel='Amplitude frequency', title=p.method,
#               cblabel='ERPAC', vmin=0., rmaxis=True)
#     plt.axvline(1., linestyle='--', color='w', linewidth=2)

# p.show()

erpac = p.fit(pha, amp, method='gc', n_perm=30)
erpac[p.pvalues > .05] = np.nan
p.pacplot(erpac.squeeze(), time, p.yvec)
p.show()