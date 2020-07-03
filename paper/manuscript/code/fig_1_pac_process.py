"""Built phase, amplitude and surrogates."""
import json
with open("../../paper.json", 'r') as f: cfg = json.load(f)  # noqa

import numpy as np

from tensorpac import Pac
from tensorpac.signals import pac_signals_tort
from brainets.utils import normalize

import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')
sns.set_style("white")
plt.rc('font', family=cfg["font"])


###############################################################################
n_perm = 5000
color_raw = '#031A16'
color_pha = '#3E7996'
color_amp = '#963E4C'
color_perm = '#818c8a'
###############################################################################

# additional functions
def rmaxis(ax, torm=['top', 'right', 'bottom', 'left']):
    """Remove axes."""
    plt.axis('tight')
    for loc, spine in ax.spines.items():
        if loc in torm:
            spine.set_color('none')
            ax.tick_params(**{loc: False})

def savefig(save_as):
    """Save the figure."""
    plt.savefig(f"../figures/Fig1_{save_as}.pdf", dpi=600, bbox_inches='tight')


# random signals
data, time = pac_signals_tort(n_epochs=1, f_pha=10, f_amp=100, sf=1024,
                              n_times=1025)
data = normalize(data, -1, 1)

# extract phase / amplitude
p = Pac(idpac=(1, 2, 3), f_pha=[9, 11], f_amp=[90, 110])
phase = p.filter(1024, data, ftype='phase')
x10hz = p.filter(1024, data, ftype='phase', keepfilt=True)
amplitude = p.filter(1024, data, ftype='amplitude', )
x100hz = p.filter(1024, data, ftype='amplitude', keepfilt=True)


###############################################################################
# Raw time-series
###############################################################################
plt.figure(1, figsize=(8.5, 2.5))
plt.plot(time, np.squeeze(data), color=color_raw, linewidth=.7)
plt.yticks([-1, 0, 1])
plt.xlabel('Time'), plt.ylabel('mV')
plt.title("Raw data")
plt.autoscale(axis='both', tight=True)
plt.tight_layout()
savefig('raw')

###############################################################################
# 10hz oscillations
###############################################################################
plt.figure(2, figsize=(8.5, 2))
plt.plot(time, np.squeeze(x10hz), color=color_pha, linewidth=2)
plt.xticks([]), plt.yticks([])
rmaxis(plt.gca())
plt.title("10hz oscillations")
savefig('10hz_osc')

###############################################################################
# 100hz oscillations
###############################################################################
plt.figure(3, figsize=(8.5, 2))
plt.plot(time, np.squeeze(x100hz), color=color_amp, linewidth=2)
plt.xticks([]), plt.yticks([])
plt.title("100hz oscillations")
rmaxis(plt.gca())
savefig('100hz_osc')

###############################################################################
# 10hz phase
###############################################################################
plt.figure(4, figsize=(8.5, 2))
plt.plot(time, np.squeeze(x10hz), color='lightgray', linewidth=1)
plt.plot(time, normalize(np.squeeze(phase), x10hz.min(), x10hz.max()),
         color=color_pha, linewidth=3)
plt.xticks([]), plt.yticks([])
plt.title("10hz phase")
rmaxis(plt.gca())
savefig('10hz_pha')

###############################################################################
# 100hz amplitude
###############################################################################
plt.figure(5, figsize=(8.5, 2))
plt.plot(time, np.squeeze(x100hz), color='lightgray', linewidth=1)
plt.plot(time, np.squeeze(amplitude), color=color_amp, linewidth=3)
plt.xticks([]), plt.yticks([])
plt.title("100hz amplitude")
rmaxis(plt.gca())
savefig('100hz_amp')

###############################################################################
# surrogates
###############################################################################
data, time = pac_signals_tort(n_epochs=10, f_pha=10, f_amp=100, sf=1024,
                              n_times=1000, rnd_state=1)
p = Pac(idpac=(1, 2, 3), f_pha=[9, 11], f_amp=[90, 110])
pha = p.filter(1024, data, ftype='phase')
amp = p.filter(1024, data, ftype='amplitude')
pac = p.fit(pha, amp, n_perm=n_perm).squeeze()
surro = p.surrogates.squeeze().max(-1)

plt.figure(figsize=(6, 4.5))
plt.hist(surro, bins=10, color=color_perm)
plt.title("Distribution of surrogates")
rmaxis(plt.gca())
plt.autoscale(axis='both', tight=True)
plt.xticks([]), plt.yticks([])
plt.tight_layout()
savefig('perm')
