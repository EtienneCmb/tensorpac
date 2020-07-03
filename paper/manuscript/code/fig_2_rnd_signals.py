"""Illustrative PAC example."""
import json
with open("../../paper.json", 'r') as f: cfg = json.load(f)  # noqa

from tensorpac.signals import pac_signals_tort, pac_signals_wavelet
from tensorpac import Pac

import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')
sns.set_style("whitegrid")
plt.rc('font', family=cfg['font'])


###############################################################################
sf = 512
f_pha, f_amp = 5, 120
n_epochs = 5
n_times = 1000
###############################################################################


# generate the random dataset
sig_t, time = pac_signals_tort(f_pha=f_pha, f_amp=f_amp, n_epochs=n_epochs,
                               noise=2, n_times=n_times, sf=sf)
sig_w, time = pac_signals_wavelet(sf=sf, f_pha=f_pha, f_amp=f_amp, noise=.5,
                                  n_epochs=n_epochs, n_times=n_times)

# compute pac
p = Pac(idpac=(6, 0, 0), f_pha=cfg["phres"], f_amp=cfg["ahres"],
        dcomplex='wavelet', width=12)
xpac_t = p.filterfit(sf, sig_t, n_jobs=1).mean(-1)
xpac_w = p.filterfit(sf, sig_w, n_jobs=1).mean(-1)

plt.figure(figsize=(17, 14))

# ----------------------------------------- Tort et al. 2010
plt.subplot(221)
plt.plot(time, sig_t[0, :], color='black', lw=.8)
plt.ylabel("Amplitude (V)")
plt.title("pac_signals_tort (Tort et al. 2010)")
plt.autoscale(axis='both', tight=True)
plt.tick_params(axis='x', which='both', labelbottom=False)
ax = plt.gca()
ax.text(*tuple(cfg["nb_pos"]), 'A', transform=ax.transAxes, **cfg["nb_cfg"])

plt.subplot(222)
p.comodulogram(xpac_t, title='Comodulogram', cmap=cfg['cmap'],
               plotas='contour', ncontours=5, fz_labels=18, fz_title=20,
               fz_cblabel=18)
plt.axvline(5, lw=1, color='white')
plt.axhline(120, lw=1, color='white')
plt.xlabel("")
plt.tick_params(axis='x', which='both', labelbottom=False)
ax = plt.gca()

# ----------------------------------------- La Tour et al. 2017
plt.subplot(223)
plt.plot(time, sig_w[0, :], color='black', lw=.8)
plt.xlabel("Time (secondes)"), plt.ylabel("Amplitude (V)")
plt.title("pac_signals_wavelet (La Tour et al. 2017)")
plt.autoscale(axis='both', tight=True)
ax = plt.gca()
ax.text(*tuple(cfg["nb_pos"]), 'B', transform=ax.transAxes, **cfg["nb_cfg"])

plt.subplot(224)
p.comodulogram(xpac_w, title='', cmap=cfg['cmap'], plotas='contour',
               ncontours=5, fz_labels=18, fz_title=20, fz_cblabel=18)
plt.axvline(5, lw=1, color='white')
plt.axhline(120, lw=1, color='white')
ax = plt.gca()

plt.tight_layout()
plt.savefig(f"../figures/Fig2.png", dpi=300, bbox_inches='tight')

plt.show()
