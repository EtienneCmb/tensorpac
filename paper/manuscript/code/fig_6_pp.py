"""Preferred phase example."""
import json
with open("../../paper.json", 'r') as f: cfg = json.load(f)  # noqa

import numpy as np

from tensorpac import PreferredPhase
from tensorpac.utils import BinAmplitude
from tensorpac.signals import pac_signals_wavelet

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')
# sns.set_style("white")
plt.rc('font', family=cfg["font"])
mpl.rcParams['grid.linewidth'] = .5
mpl.rcParams['grid.color'] = "k"
mpl.rcParams['ytick.color'] = "k"
mpl.rcParams['ytick.labelsize'] = "large"

###############################################################################
sf = 1024.
n_epochs = 100
n_times = 2000
pp = np.pi / 4
###############################################################################
data, time = pac_signals_wavelet(f_pha=6, f_amp=100, n_epochs=n_epochs, sf=sf,
                                 noise=3, n_times=n_times, pp=pp)

# compute the binned amplitude
b_obj = BinAmplitude(data, sf, f_pha=[5, 7], f_amp=[90, 110], n_jobs=1,
                     n_bins=18)

# compute the preferred phase
p = PreferredPhase(f_pha=[5, 7], f_amp=(60, 200, 10, 1))
pha = p.filter(sf, data, ftype='phase', n_jobs=1)
amp = p.filter(sf, data, ftype='amplitude', n_jobs=1)
ampbin, pp, vecbin = p.fit(pha, amp, n_bins=72)
pp = np.squeeze(pp).T
ampbin = np.squeeze(ampbin).mean(-1)

plt.figure(figsize=(18, 8))

plt.subplot(121)
ax = b_obj.plot(color='#34495e', alpha=.5, unit='deg')
plt.ylim(40, 140)
plt.title("Binned 100hz amplitude according to the 6hz phase")
# plt.autoscale(tight=True)
ax = plt.gca()
ax.text(*tuple(cfg["nb_pos"]), 'A', transform=ax.transAxes, **cfg["nb_cfg"])

plt.subplot(122)
ax = p.polar(ampbin.T, vecbin, p.yvec, cmap="RdBu_r", interp=.1,
             cblabel='Amplitude bins', subplot=122, fz_cblabel=18)
ax.set_rlabel_position(-45)
ax = plt.gca()
ax.text(*tuple(cfg["nb_pos"]), 'B', transform=ax.transAxes, **cfg["nb_cfg"])
# plt.title("Preferred Phase of 45° of a 6hz <-> 100hz coupling")

plt.tight_layout()
plt.savefig(f"../figures/Fig6.png", dpi=300, bbox_inches='tight')

p.show()
