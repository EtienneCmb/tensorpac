"""Performance comparison between the Tensor / Numba implementations."""
import json
with open("../../paper.json", 'r') as f: cfg = json.load(f)  # noqa

import numpy as np
import pandas as pd
from time import time as tst

from tensorpac.signals import pac_signals_wavelet
from tensorpac import Pac
from tensorpac.spectral import hilbertm
from tensorpac.methods import get_pac_fcn

import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')
sns.set_style("white")
plt.rc('font', family=cfg["font"])


###############################################################################
n_repetitions = 10
f_pha, f_amp = 10, 100
n_epochs, n_times = 30, 2000
sf = 512.
###############################################################################

# -----------------------------------------------------------------------------
# get tensor / numba implemented PAC methods
# -----------------------------------------------------------------------------

# first run could be slow due to numba caching
METH_TENSOR = get_pac_fcn(None, 18, .05, 'tensor', full=True)
METH_NUMBA = get_pac_fcn(None, 18, .05, 'numba', full=True)
# remove gaussian copula since there's no Numba implementation
METH_TENSOR.pop(6)
METH_NUMBA.pop(6)
n_meth = len(METH_TENSOR) + len(METH_NUMBA)

# -----------------------------------------------------------------------------
# DATA PREPARATION
# -----------------------------------------------------------------------------

# generate random data
data, time = pac_signals_wavelet(sf=sf, f_pha=f_pha, f_amp=f_amp, noise=.8,
                                 n_epochs=n_epochs, n_times=n_times)
# extract phase / amplitude
p_obj = Pac(f_pha='lres', f_amp='lres')
pha = p_obj.filter(sf, data, ftype='phase', n_jobs=1)
amp = p_obj.filter(sf, data, ftype='amplitude', n_jobs=1)

# -----------------------------------------------------------------------------
# compute pac
# -----------------------------------------------------------------------------

meth_types, meth_names, computing_time = [], [], []
for meth in range(len(METH_TENSOR)):
    # get both methods
    meth_tensor = METH_TENSOR[meth + 1]
    meth_numba = METH_NUMBA[meth + 1]
    # get both names
    meth_tensor_name = meth_tensor.func.__name__
    meth_numba_name = meth_numba.func.__name__
    assert meth_numba_name == f"{meth_tensor_name}_nb"
    # compute several repetitions
    for rep in range(n_repetitions):
        # compute tensor based
        start_ten = tst()
        meth_tensor(pha, amp)
        end_ten = tst()
        computing_time += [end_ten - start_ten]
        meth_names += [meth_tensor_name]
        meth_types += ['Tensor']
        # compute numba based
        start_nb = tst()
        meth_numba(pha, amp)
        end_nb = tst()
        computing_time += [end_nb - start_nb]
        meth_names += [meth_tensor_name]
        meth_types += ['Numba']

# -----------------------------------------------------------------------------
# build DataFrame and barplot the results
# -----------------------------------------------------------------------------

df = pd.DataFrame({'Computing time': computing_time, 'Method': meth_names,
                  'Implementation': meth_types})

plt.figure(figsize=(10, 9))
sns.barplot(x="Method", y="Computing time", hue="Implementation", data=df)
plt.title("Computing time comparison between the tensor and numba "
          f"implementations\n(n_pha={len(p_obj.xvec)}, "
          f"n_amp={len(p_obj.yvec)}, n_trials={n_epochs}, n_times={n_times})")
plt.xticks(rotation=10)
plt.tight_layout()

plt.savefig(f"../figures/r2_tensor_vs_numba.png", dpi=300, bbox_inches='tight')

plt.show()
