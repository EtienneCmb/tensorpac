"""In this example we illustrate how to compute basic phase amplitude coupling.

For the visualization, we used a comodulogram.
"""
import matplotlib.pyplot as plt
from tensorpac.utils import PacSignals
from tensorpac import Pac
plt.style.use('seaborn-paper')

# First, we generate a dataset of signals artificially coupled between 10hz
# and 100hz. By default, this dataset is organized as (ndatasets, npts) where
# npts is the number of time points.
n = 100  # number of datasets
data, time = PacSignals(fpha=10, famp=100, noise=3, ndatasets=n, dpha=10, damp=10)

# First, let's use the MVL, without any further correction by surrogates :
p = Pac(idpac=(4, 0, 0), fpha=(2, 30, 1, 1), famp=(60, 150, 5, 5),
        dcomplex='wavelet', width=12)
xpac, pval = p.filterfit(1024, data, data, axis=1, nperm=210)

p.comodulogram(xpac.mean(-1), title='Contour plot with 5 regions',
               cmap='Spectral_r', plotas='contour', ncontours=5, vmin=60,
               vmax=300, rmaxis=True)

# plt.show()
plt.savefig('tp4.png', bbox_inches='tight', format='png', dpi=600)