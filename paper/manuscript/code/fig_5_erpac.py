"""Event Related PAC."""
import json
with open("../../paper.json", 'r') as f: cfg = json.load(f)  # noqa

import numpy as np

from tensorpac import EventRelatedPac
from tensorpac.signals import pac_signals_wavelet

import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')
sns.set_style("white")
plt.rc('font', family=cfg["font"])

###############################################################################
n_epochs = 300
n_times = 1000
sf = 1000.
###############################################################################

x1, tvec = pac_signals_wavelet(f_pha=10, f_amp=100, n_epochs=n_epochs, noise=2,
                               n_times=n_times, sf=sf)
x2 = np.random.rand(n_epochs, 1000)
x = np.concatenate((x1, x2), axis=1)
time = np.arange(x.shape[1]) / sf


p = EventRelatedPac(f_pha=[9, 11], f_amp=cfg["ahres"])
# p = EventRelatedPac(f_pha=[9, 11], f_amp='lres')
pha = p.filter(sf, x, ftype='phase', n_jobs=-1)
amp = p.filter(sf, x, ftype='amplitude', n_jobs=-1)

plt.figure(figsize=(16, 6))
for n_m, (method, nb) in enumerate(zip(['circular', 'gc'], ['A', 'B'])):
    # to be fair with the comparison between ERPAC and gcERPAC, the smoothing
    # parameter of the gcERPAC but results could look way better if for
    # example with add a `smooth=20`
    erpac = p.fit(pha, amp, method=method, n_jobs=-1).squeeze()
    plt.subplot(1, 2, n_m + 1)
    p.pacplot(erpac, time, p.yvec, xlabel='Time (second)', cmap=cfg["cmap"],
              ylabel='Frequency for amplitude (Hz)', title=p.method,
              vmin=0., rmaxis=True, fz_labels=20, fz_title=22, fz_cblabel=20)
    plt.axvline(1., linestyle='--', color='w', linewidth=2)
    if n_m == 1: plt.ylabel('')
    ax = plt.gca()
    ax.text(*tuple(cfg["nb_pos"]), nb, transform=ax.transAxes, **cfg["nb_cfg"])

plt.tight_layout()
plt.savefig(f"../figures/Fig5.png", dpi=300, bbox_inches='tight')

plt.show()
