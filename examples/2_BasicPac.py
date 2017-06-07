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
p = Pac(idpac=(1, 0, 0), fpha=(2, 30, 2, 1), famp=(60, 150, 10, 5))
xpac, _ = p.filterfit(1024, data, data, axis=1)

# Now, we still use the MVL method, but in addition we shuffle amplitude time
# series and then, subtract then divide by the mean of surrogates :
p.idpac = (1, 3, 3)
xpac_corr, _ = p.filterfit(1024, data, data, axis=1, nperm=10)

# Now, we plot the result by taking the mean across the dataset dimension.
plt.subplot(1, 2, 1)
p.comodulogram(xpac.mean(-1), title='Without surrogate correction')

plt.subplot(1, 2, 2)
p.comodulogram(xpac_corr.mean(-1), title='With surrogate correction')
plt.show()