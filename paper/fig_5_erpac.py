"""Event Related PAC."""
import json
with open("paper.json", 'r') as f: cfg = json.load(f)  # noqa

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


p = EventRelatedPac(f_pha=[9, 11], f_amp=(60, 140, 5, 1))
pha = p.filter(sf, x, ftype='phase', n_jobs=1)
amp = p.filter(sf, x, ftype='amplitude', n_jobs=1)

plt.figure(figsize=(8, 6))
erpac = p.fit(pha, amp, method='circular', n_jobs=-1).squeeze()
p.pacplot(erpac, time, p.yvec, xlabel='Time (second)', cmap=cfg["cmap"],
          ylabel='Frequency for amplitude (Hz)', title='Event Related PAC',
          vmin=0., rmaxis=True)
plt.axvline(1., linestyle='--', color='w', linewidth=2)

plt.tight_layout()
plt.savefig(f"{cfg['path']}/Fig5.png", dpi=300, bbox_inches='tight')

plt.show()
