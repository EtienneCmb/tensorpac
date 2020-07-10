"""Comparison between pactools and tensoprpac."""
import json
with open("../../paper.json", 'r') as f: cfg = json.load(f)  # noqa

import pandas as pd
import numpy as np
from time import time as tst

from tensorpac.signals import pac_signals_wavelet
from tensorpac import Pac

from pactools import Comodulogram, simulate_pac

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.style.use('seaborn-poster')
sns.set_style("white")
plt.rc('font', family=cfg["font"])


###############################################################################
# simulated data parameters
sf = 512.
n_epochs = 20
n_times = 4000
n_perm = 20
# frequency vectors resolutions
n_pha = 50
n_amp = 40
# method correspondance between pactools and tensorpac
METH = dict(
        PACTOOLS=dict(MVL='canolty', MI='tort', PLV='penny'),
        TENSORPAC=dict(MVL=1, MI=2, PLV=5)
    )
###############################################################################

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                       SIMULATE PAC + FREQUENCY VECTORS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
data, time = pac_signals_wavelet(sf=sf, f_pha=10, f_amp=100, noise=.8,
                              n_epochs=n_epochs, n_times=n_times)

# construct frequency vectors that fit both tensorpac and pactools
f_pha_pt_width = 1.
f_pha_pt = np.linspace(3, 20, n_pha)
f_pha_tp = np.c_[f_pha_pt - f_pha_pt_width / 2, f_pha_pt + f_pha_pt_width / 2]

f_amp_pt_width = max(f_pha_pt) * 2  # pactools recommandation
f_amp_pt = np.linspace(60, 140, n_amp)
f_amp_tp = np.c_[f_amp_pt - f_amp_pt_width / 2, f_amp_pt + f_amp_pt_width / 2]


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                            COMPUTING FUNCTION
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def compute_pt_tp(n_jobs, n_perm, title=''):
    """Compute Pactools and Tensorpac."""
    cpt, meth_comp, soft = [], [], []
    for meth in ['MVL', 'MI', 'PLV']:
        # ---------------------------------------------------------------------
        # get the method name
        meth_pt = METH['PACTOOLS'][meth]
        meth_tp = METH['TENSORPAC'][meth]

        if n_perm > 0:
            idpac = (meth_tp, 2, 4)
        else:
            idpac = (meth_tp, 0, 0)

        # ---------------------------------------------------------------------
        # PACTOOLS
        pt_start = tst()
        estimator = Comodulogram(
            fs=sf, low_fq_range=f_pha_pt, low_fq_width=f_pha_pt_width,
            high_fq_range=f_amp_pt, high_fq_width=f_amp_pt_width,
            method=meth_pt, progress_bar=False, n_jobs=n_jobs,
            n_surrogates=n_perm)
        estimator.fit(data)
        pt_end = tst()
        # ---------------------------------------------------------------------
        # TENSORPAC
        tp_start = tst()
        p_obj = Pac(idpac=idpac, f_pha=f_pha_tp, f_amp=f_amp_tp,
                    verbose='error')
        pac = p_obj.filterfit(sf, data, n_jobs=n_jobs, n_perm=n_perm).mean(-1)
        tp_end = tst()

        # ---------------------------------------------------------------------
        # shape shape checking
        assert estimator.comod_.shape == pac.T.shape
        # saving the results
        cpt += [pt_end - pt_start, tp_end - tp_start]
        meth_comp += [meth] * 2
        soft += ['Pactools', 'Tensorpac']

    # -------------------------------------------------------------------------
    df = pd.DataFrame({"Computing time": cpt, "PAC method": meth_comp,
                      "Software": soft})
    sns.barplot(x="PAC method", y="Computing time", hue="Software", data=df)
    plt.title(title, fontsize=14, fontweight='bold')


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                            RUN THE COMPARISON
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

plt.figure(figsize=(12, 10))

# single-core // no perm
plt.subplot(2, 2, 1)
compute_pt_tp(1, 0, title="Single-core // no surrogates")
plt.subplot(2, 2, 2)
compute_pt_tp(-1, 0, title="Multi-core // no surrogates")
plt.subplot(2, 2, 3)
compute_pt_tp(1, n_perm, title=f"Single-core // {n_perm} surrogates")
plt.subplot(2, 2, 4)
compute_pt_tp(-1, n_perm, title=f"Multi-core // {n_perm} surrogates")

plt.tight_layout()
plt.savefig(f"../figures/r3_pactools_vs_tensorpac.png", dpi=300,
            bbox_inches='tight')

plt.show()
