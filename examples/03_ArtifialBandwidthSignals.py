"""This example illustrate that it's also possible to generate a coupling
between specific bands (i.e. [5, 7] <-> [60, 80] hz).
"""
import numpy as np
import matplotlib.pyplot as plt

from tensorpac.utils import PacSignals
from tensorpac import Pac

data, time = PacSignals(fpha=[5, 7], famp=[60, 80], chi=0.5, ndatasets=50,
                        noise=3., npts=2000)


p = Pac(idpac=(3, 1, 1), fpha=(1, 15, 1, .2), famp=(40, 100, 5, 2),
        dcomplex='wavelet', width=6)
pac = np.squeeze(p.filterfit(1024, data, data, axis=1, nperm=10)[0])

p.comodulogram(pac.mean(-1), vmin=0., title=str(p))
plt.show()
