"""[f_min, f_max] optimisation"""
import json
with open("paper.json", 'r') as f: cfg = json.load(f)  # noqa

from tensorpac import Pac
from tensorpac.signals import pac_signals_tort, pac_signals_wavelet
from tensorpac.utils import pac_trivec, PSD

import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')
sns.set_style("white")
plt.rc('font', family=cfg["font"])

###############################################################################
sf = 256.
n_times = 3000
n_epochs = 20
###############################################################################
data, time = pac_signals_wavelet(f_pha=6, f_amp=70, n_epochs=n_epochs, noise=1,
                                 n_times=n_times, sf=sf)

psd = PSD(data, sf)

trif, tridx = pac_trivec(f_start=40, f_end=100, f_width=3)

p = Pac(idpac=(6, 0, 0), f_pha=[5, 7], f_amp=trif)
pha = p.filter(sf, data, ftype='phase', n_jobs=1)
amp = p.filter(sf, data, ftype='amplitude', n_jobs=1)
pac = p.fit(pha, amp, n_jobs=-1).mean(-1).squeeze()

best_f = trif[pac.argmax()]

plt.figure(figsize=(16, 7))
plt.subplot(121)
ax = psd.plot(confidence=None, f_min=2, f_max=30, log=False, grid=True)
plt.ylim(0, .15)
plt.title("Power Spectrum Density (PSD)")
# plt.autoscale(enable=True, axis='y', tight=True)
ax = plt.gca()
ax.text(*tuple(cfg["nb_pos"]), 'A', transform=ax.transAxes, **cfg["nb_cfg"])

plt.subplot(122)
p.triplot(pac, trif, tridx, cmap=cfg["cmap"], rmaxis=True,
          title=r'Optimal $[Fmin; Fmax]hz$ band for amplitude', cblabel="")
plt.axvline(best_f[0], lw=1, color='w')
plt.axhline(best_f[1], lw=1, color='w')
ax = plt.gca()
ax.text(*tuple(cfg["nb_pos"]), 'B', transform=ax.transAxes, **cfg["nb_cfg"])

plt.tight_layout()
plt.savefig(f"{cfg['path']}/Fig7.png", dpi=300, bbox_inches='tight')

print("*" * 79)
print(f"BEST FREQUENCY RANGE : {trif[pac.argmax()]}")
print("*" * 79)

p.show()
