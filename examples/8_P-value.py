"""In this example we illustrate how to compute basic phase amplitude coupling.

For the visualization, we used a comodulogram.
"""
import matplotlib.pyplot as plt
from tensorpac.utils import PacSignals
from tensorpac import Pac
plt.style.use('seaborn-poster')

# First, we generate a dataset of signals artificially coupled between 10hz
# and 100hz. By default, this dataset is organized as (ndatasets, npts) where
# npts is the number of time points.
n = 100  # number of datasets
data, time = PacSignals(fpha=10, famp=100, noise=0, ndatasets=n)

# First, let's use the MVL, without any further correction by surrogates :
p = Pac(idpac=(1, 3, 3), fpha=(2, 30, 2, 2), famp=(60, 150, 10, 10),
        dcomplex='wavelet', width=12)
xpac, pval = p.filterfit(1024, data, data, axis=1, nperm=100)
t1 = p.method + '\n' + p.surro + '\n' + p.norm

# Now, we plot the result by taking the mean across the dataset dimension.
p.comodulogram(xpac.mean(-1), title=t1, cmap='Spectral_r', vmin=.2,
               pvalues=pval.mean(-1), bad='orange')

plt.show()