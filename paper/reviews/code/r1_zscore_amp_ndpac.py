"""#R1 : assumptions of the ndPAC.

ndPAC make the assumptions that the phase is normally distributed and the
z-score amplitude is normally distributed. This script illustrate how those
assumptions hold.

"""
import json
with open("../../paper.json", 'r') as f: cfg = json.load(f)  # noqa

from tensorpac.signals import pac_signals_tort, pac_signals_wavelet
from tensorpac import Pac

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.style.use('seaborn-poster')
sns.set_style("white")
plt.rc('font', family=cfg["font"])


###############################################################################
sf = 1024
n_epochs = 1
n_times = 100000
n_bins = 30
###############################################################################

data, time = pac_signals_wavelet(sf=sf, f_pha=10, f_amp=100, noise=2.,
                                 n_epochs=n_epochs, n_times=n_times,
                                 rnd_state=0)


# extract the phase and the amplitude
p = Pac(idpac=(1, 0, 0), f_pha=[8, 12], f_amp=[60, 140])
pha = p.filter(sf, data, ftype='phase', n_jobs=1).squeeze()
amp = p.filter(sf, data, ftype='amplitude', n_jobs=1).squeeze()

# z-score normalize the amplitude
amp_n = (amp - amp.mean()) / amp.std()


plt.figure(figsize=(18, 5))

plt.subplot(131)
plt.hist(pha, n_bins, color='blue')
plt.title('Binned phase')

plt.subplot(132)
plt.hist(amp, n_bins, color='red')
plt.title('Binned amplitude')

plt.subplot(133)
plt.hist(amp_n, n_bins, color='orange')
plt.title('Binned z-scored amplitude')


plt.savefig(f"../figures/r1_ndpac_assumptions.png", dpi=300,
            bbox_inches='tight')

plt.show()