"""Reproduction of the test_functional_pac."""
import json
with open("../../paper.json", 'r') as f: cfg = json.load(f)  # noqa

import numpy as np
from scipy import stats

from tensorpac import Pac
from tensorpac.signals import pac_signals_wavelet

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-poster')
sns.set_style("white")
plt.rc('font', family=cfg["font"])

# generate a 10<->100hz ground truth coupling
n_epochs, sf, n_times = 1, 512., 4000
data, time = pac_signals_wavelet(f_pha=10, f_amp=100, noise=.8,
                                 n_epochs=n_epochs, n_times=n_times,
                                 sf=sf)
# phase / amplitude extraction (single time)
p = Pac(f_pha='lres', f_amp='lres', dcomplex='wavelet', width=12)
phases = p.filter(sf, data, ftype='phase', n_jobs=1)
amplitudes = p.filter(sf, data, ftype='amplitude', n_jobs=1)
# ground truth array construction
n_pha, n_amp = len(p.xvec), len(p.yvec)
n_pix = int(n_pha * n_amp)
gt = np.zeros((n_amp, n_pha), dtype=bool)
b_pha = np.abs(p.xvec.reshape(-1, 1) - np.array([[9, 11]])).argmin(0)
b_amp = np.abs(p.yvec.reshape(-1, 1) - np.array([[95, 105]])).argmin(0)
gt[b_amp[0]:b_amp[1] + 1, b_pha[0]:b_pha[1] + 1] = True

plt.figure(figsize=(12, 9))
plt.subplot(2, 3, 1)
p.comodulogram(gt, title='Gound truth', cmap='magma', colorbar=False)
# loop over implemented methods
for i, k in enumerate([1, 2, 3, 5, 6]):
    # compute only the pac
    p.idpac = (k, 2, 3)
    xpac = p.fit(phases, amplitudes, n_perm=200).squeeze()
    pval = p.pvalues.squeeze()
    is_coupling = pval <= .05
    # count the number of correct pixels. This includes both true
    # positives and true negatives
    trpr = (is_coupling == gt).sum() / n_pix
    assert trpr > .95
    # build title of the figure (for sanity check)
    meth = p.method.replace(' (', '\n(')
    title = f"Method={meth}\nAccuracy={np.around(trpr * 100, 2)}%"
    # set to nan everywhere it's not significant
    xpac[~is_coupling] = np.nan
    vmin, vmax = np.nanmin(xpac), np.nanmax(xpac)
    # plot the results
    plt.subplot(2, 3, i + 2)
    p.comodulogram(xpac, colorbar=False, vmin=vmin, vmax=vmax, title=title)
    plt.ylabel(''), plt.xlabel('')
plt.tight_layout()

plt.savefig(f"../figures/r2_functional_pac.png", dpi=300, bbox_inches='tight')

plt.show()  # show on demand