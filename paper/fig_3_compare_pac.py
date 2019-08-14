"""Compare PAC methods."""
import json
with open("paper.json", 'r') as f: cfg = json.load(f)  # noqa

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
n_epochs = 20
n_times = 3000
n_perm = 20
titles = ["MVL", "KLD", "HR", "ndPAC", "PS"]
methods = dict(MVL=[0, slice(0, 2)], KLD=[0, slice(2, 4)],
               HR=[0, slice(4, 6)], ndPAC=[1, slice(1, 3)],
               PS=[1, slice(3, 5)])
###############################################################################

data, time = pac_signals_wavelet(sf=sf, f_pha=10, f_amp=100, noise=2.,
                                 n_epochs=n_epochs, n_times=n_times)


p = Pac(idpac=(1, 2, 3), f_pha=cfg["phres"], f_amp=cfg["ahres"])
# p = Pac(idpac=(1, 2, 3), f_pha=(1, 30, 1, 1), f_amp=(60, 160, 5, 5))
# p = Pac(idpac=(1, 2, 3), f_pha='mres', f_amp='mres')
pha = p.filter(sf, data, ftype='phase', n_jobs=1)
amp = p.filter(sf, data, ftype='amplitude', n_jobs=1)

plt.figure(figsize=(18, 10))
gs = GridSpec(2, 7)

for k in range(5):
    p.idpac = (k + 1, 2, 3)
    xpac = p.fit(pha.copy(), amp.copy(), n_perm=n_perm, p=.05,
                 n_jobs=-1).mean(-1)

    sq = methods[titles[k]]
    plt.subplot(gs[sq[0], sq[1]])
    p.comodulogram(xpac, cblabel="", title=titles[k], cmap=cfg["cmap"],
                   vmin=0, colorbar=True)
    if k > 0:
        plt.ylabel("")
        # plt.tick_params(axis='y', which='both', labelleft=False)

plt.tight_layout()
plt.savefig(f"{cfg['path']}/Fig3.png", dpi=300,
            bbox_inches='tight')

p.show()