"""#R1 : threshold of the ndPAC.

This script provide an insight of the ndPAC's threshold.
"""
import json
with open("../../paper.json", 'r') as f: cfg = json.load(f)  # noqa

import numpy as np
from scipy.special import erfinv

from tensorpac.signals import pac_signals_wavelet
from tensorpac import Pac

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.style.use('seaborn-poster')
sns.set_style("white")
plt.rc('font', family=cfg["font"])



def custom_ndpac(pha, amp, p=.05):
    npts = amp.shape[-1]
    # Normalize amplitude :
    # Use the sample standard deviation, as in original Matlab code from author
    amp = np.subtract(amp, np.mean(amp, axis=-1, keepdims=True))
    amp = np.divide(amp, np.std(amp, ddof=1, axis=-1, keepdims=True))
    # Compute pac :
    pac = np.abs(np.einsum('i...j, k...j->ik...', amp, np.exp(1j * pha)))

    s = pac ** 2
    pac /= npts
    # Set to zero non-significant values:
    xlim = npts * erfinv(1 - p)**2

    pac_nt = pac.copy()
    pac[s <= 2 * xlim] = np.nan

    return pac_nt.squeeze(), pac.squeeze(), s.squeeze()




if __name__ == '__main__':
    ###########################################################################
    p = .05
    sf = 1024
    n_epochs = 1
    n_times = 100000
    ###########################################################################

    data, time = pac_signals_wavelet(sf=sf, f_pha=10, f_amp=100, noise=2.,
                                     n_epochs=n_epochs, n_times=n_times)


    # extract the phase and the amplitude
    p_obj = Pac(idpac=(1, 0, 0), f_pha=cfg["phres"], f_amp=cfg["ahres"])
    # p_obj = Pac(idpac=(1, 0, 0), f_pha='mres', f_amp='mres')
    pha = p_obj.filter(sf, data, ftype='phase', n_jobs=1)
    amp = p_obj.filter(sf, data, ftype='amplitude', n_jobs=1)

    # compute PAC (outside of the PAC object)
    pac_nt, pac, s = custom_ndpac(pha, amp, p=p)

    plt.figure(figsize=(22, 6))
    plt.subplot(131)
    p_obj.comodulogram(pac_nt, cblabel="", title='Non-thresholded PAC',
                       cmap=cfg["cmap"], vmin=0, colorbar=True)
    plt.subplot(132)
    p_obj.comodulogram(s, cblabel="", title='Threshold', cmap=cfg["cmap"],
                       vmin=0, colorbar=True)
    plt.ylabel('')
    plt.subplot(133)
    p_obj.comodulogram(pac, cblabel="", title='Thresholded PAC',
                       cmap=cfg["cmap"], vmin=0, colorbar=True)
    plt.ylabel('')
    plt.tight_layout()

    plt.savefig(f"../figures/r1_ndpac_threshold.png", dpi=300,
                bbox_inches='tight')

    plt.show()