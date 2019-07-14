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
p = EventRelatedPac(f_pha=[9, 11], f_amp=(60, 140, 5, 5))

# extract phases and amplitudes
erpac = p.filterfit(sf, x, method='gc', n_perm=20).squeeze()

# erpac = p.pvalues.squeeze()
surro = p.surrogates.squeeze()
print(surro.shape)

plt.subplot(131)
plt.hist(erpac.ravel(), color='red', alpha=.5)
plt.hist(surro.ravel(), color='blue', alpha=.5)
plt.subplot(132)
plt.pcolormesh(erpac)
plt.subplot(133)
plt.pcolormesh(surro.mean(0))
plt.show()
0/0
# erpac[pvalues > .05] = np.nan


p.pacplot(erpac, time, p.yvec, xlabel='Time (second)',
          cmap='Spectral_r', ylabel='Amplitude frequency', title=p.method,
          cblabel='ERPAC', vmin=0., rmaxis=True)
plt.axvline(1., linestyle='--', color='w', linewidth=2)

p.show()
