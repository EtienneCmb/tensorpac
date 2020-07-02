"""Compare single trial vs tensor computations."""
import json
with open("paper.json", 'r') as f: cfg = json.load(f)  # noqa

import numpy as np
import pandas as pd
from time import time as tst

from tensorpac.signals import pac_signals_tort, pac_signals_wavelet
from tensorpac.methods import (mvl, kld, hr, ndpac, ps)
from tensorpac import Pac


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.style.use('seaborn-poster')
sns.set_style("white")
plt.rc('font', family=cfg["font"])


###############################################################################
sf = 1024
n_epochs = 100
n_times = 3000
res = 'hres'
titles = ["MVL", "KLD", "HR", "ndPAC", "PS"]
methods = dict(MVL=[0, slice(0, 2)], KLD=[0, slice(2, 4)],
               HR=[0, slice(4, 6)], ndPAC=[1, slice(1, 3)],
               PS=[1, slice(3, 5)])
###############################################################################

METHODS = dict(MVL=mvl, KLD=kld, HR=hr, ndPAC=ndpac, PS=ps)

data, time = pac_signals_wavelet(sf=sf, f_pha=10, f_amp=100, noise=2.,
                                 n_epochs=n_epochs, n_times=n_times)
# p = Pac(idpac=(1, 0, 0), f_pha=cfg["phres"], f_amp=cfg["ahres"])
# p = Pac(idpac=(1, 0, 0), f_pha=(1, 30, 1, 1), f_amp=(60, 160, 5, 5))
p = Pac(idpac=(1, 0, 0), f_pha=res, f_amp=res)
pha = p.filter(sf, data, ftype='phase', n_jobs=1)
amp = p.filter(sf, data, ftype='amplitude', n_jobs=1)
n_pha, n_amp = pha.shape[0], amp.shape[0]

print(f'Vector-based dimensions : ({n_times}, {n_epochs}, {n_pha}, {n_amp})')

elapsed, name, ctype, ratio, ratio_names = [], [], [], [], []
for n_m, (meth_name, meth) in enumerate(METHODS.items()):
    # tensor-based computations
    t_start = tst()
    meth(pha, amp)
    t_end = tst()
    t_tensor = t_end - t_start
    elapsed += [t_tensor]
    name += [meth_name]
    ctype += ['tensor']
    # vector-based computations
    t_start = tst()
    for t in range(n_epochs):
        for p in range(n_pha):
            for a in range(n_amp):
                meth(pha[[p], [t], :], amp[[a], [t], :])
    t_end = tst()
    t_vector = t_end - t_start
    elapsed += [t_vector]
    name += [meth_name]
    ctype += ['vector']

    ratio += [t_vector / t_tensor]
    ratio_names += [meth_name]

df = pd.DataFrame({"Method": name, "Computing time (s)": elapsed,
                   "Implementation": ctype})
df_ratio = pd.DataFrame({"Computing time ratio": ratio, "Method": ratio_names})

plt.figure(figsize=(16, 7))

plt.subplot(121)
sns.barplot(x="Method", y="Computing time (s)", hue="Implementation", data=df,
            palette=["#2ecc71", "#e74c3c"])
plt.grid()
plt.title("Computing time comparison\n(vector vs. tensor)")
ax = plt.gca()
ax.text(*tuple(cfg["nb_pos"]), 'A', transform=ax.transAxes, **cfg["nb_cfg"])

plt.subplot(122)
sns.barplot(x="Method", y="Computing time ratio", data=df_ratio,
            palette=["#34495e"])
plt.grid()
plt.title("Computing time ratio\n(vector / tensor)")
ax = plt.gca()
ax.text(*tuple(cfg["nb_pos"]), 'B', transform=ax.transAxes, **cfg["nb_cfg"])

plt.savefig(f"{cfg['path']}/Fig8_{res}.png", dpi=300, bbox_inches='tight')

plt.show()
