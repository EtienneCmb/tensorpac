"""Illustrative example of the false positives and false negatives ERPAC."""
import json
with open("../../paper.json", 'r') as f: cfg = json.load(f)  # noqa

import numpy as np
from scipy import stats
from mne.stats import bonferroni_correction

from tensorpac.utils import pac_trivec
from tensorpac.signals import pac_signals_wavelet
from tensorpac import EventRelatedPac

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-poster')
sns.set_style("white")
plt.rc('font', family=cfg["font"])

# erpac simultation
n_epochs, n_times, sf, edges = 400, 1000, 512., 50
x, times = pac_signals_wavelet(f_pha=10, f_amp=100, n_epochs=n_epochs,
                               noise=.1, n_times=n_times, sf=sf)
times = times[edges:-edges]
# phase / amplitude extraction (single time)
p = EventRelatedPac(f_pha=[8, 12], f_amp=(30, 200, 5, 5),
                    dcomplex='wavelet', width=12)
kw = dict(n_jobs=1, edges=edges)
phases = p.filter(sf, x, ftype='phase', **kw)
amplitudes = p.filter(sf, x, ftype='amplitude', **kw)
n_amp = len(p.yvec)
# generate a normal distribution
gt = np.zeros((n_amp, n_times - 2 * edges))
b_amp = np.abs(p.yvec.reshape(-1, 1) - np.array([[80, 120]])).argmin(0)
gt[b_amp[0]:b_amp[1] + 1, :] = True

plt.figure(figsize=(16, 5))
plt.subplot(131)
p.pacplot(gt, times, p.yvec, title='Ground truth', cmap='magma')

for n_meth, meth in enumerate(['circular', 'gc']):
    # compute erpac + p-values
    erpac = p.fit(phases, amplitudes, method=meth,
                  mcp='bonferroni', n_perm=30).squeeze()
    pvalues = p.pvalues.squeeze()
    # find everywhere erpac is significant + compare to ground truth
    is_signi = pvalues < .05
    erpac[~is_signi] = np.nan
    # computes accuracy
    acc = 100 * (is_signi == gt).sum() / (n_amp * n_times)
    assert acc > 80.
    # plot the result
    title = f"Method={p.method}\nAccuracy={np.around(acc, 2)}%"
    plt.subplot(1, 3, n_meth + 2)
    p.pacplot(erpac, times, p.yvec, title=title)
plt.tight_layout()

plt.savefig(f"../figures/r3_functional_erpac.png", dpi=300,
            bbox_inches='tight')

plt.show()
