"""Rectify PAC estimation by surrogate distribution."""
import json
with open("../../paper.json", 'r') as f: cfg = json.load(f)  # noqa

from tensorpac.signals import pac_signals_tort, pac_signals_wavelet
from tensorpac import Pac

import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')
sns.set_style("white")
plt.rc('font', family=cfg["font"])


###############################################################################
sf = 1024
n_epochs = 50
n_times = 3000
n_perm = 20
###############################################################################

data, time = pac_signals_wavelet(sf=sf, f_pha=10, f_amp=100, noise=3.,
                                 n_epochs=n_epochs, n_times=n_times)


p = Pac(idpac=(1, 2, 3), f_pha=cfg["phres"], f_amp=cfg["ahres"])
# p = Pac(idpac=(1, 2, 3), f_pha='lres', f_amp='lres')
xpac = p.filterfit(sf, data, n_perm=n_perm, p=.05, n_jobs=-1)

pacn = p.pac.mean(-1)
pac = xpac.mean(-1)
surro = p.surrogates.mean(0).max(-1)  # mean(perm).max(epochs)

kw_plt = dict(fz_labels=20, fz_title=22, fz_cblabel=20)

plt.figure(figsize=(22, 6))

plt.subplot(1, 3, 1)
p.comodulogram(pacn, cblabel="", title="Uncorrected PAC", cmap=cfg["cmap"],
               vmin=0, **kw_plt)
ax = plt.gca()
ax.text(*tuple(cfg["nb_pos"]), 'A', transform=ax.transAxes, **cfg["nb_cfg"])

plt.subplot(1, 3, 2)
p.comodulogram(surro, cblabel="", title="Mean of the surrogate\ndistribution",
               cmap=cfg["cmap"], vmin=0, **kw_plt)
plt.ylabel("")
# plt.tick_params(axis='y', which='both', labelleft=False)
ax = plt.gca()
ax.text(*tuple(cfg["nb_pos"]), 'B', transform=ax.transAxes, **cfg["nb_cfg"])

plt.subplot(1, 3, 3)
p.comodulogram(pac, cblabel="", title="Corrected PAC", cmap=cfg["cmap"],
               vmin=0, **kw_plt)
plt.ylabel("")
# plt.tick_params(axis='y', which='both', labelleft=False)
ax = plt.gca()
ax.text(*tuple(cfg["nb_pos"]), 'C', transform=ax.transAxes, **cfg["nb_cfg"])


plt.tight_layout()
plt.savefig(f"../figures/Fig4.png", dpi=300, bbox_inches='tight')

p.show()