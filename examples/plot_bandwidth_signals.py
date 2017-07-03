"""
==========================================
Generate a coupling between specific bands
==========================================

Alternativerly, you can create coupling not between centered frequences, but
between frequency bands. In this example, we illustrate a [5, 7]<->[60, 80]hz
coupling.
"""
import numpy as np
import matplotlib.pyplot as plt

from tensorpac.utils import pac_signals
from tensorpac import Pac

data, time = pac_signals(fpha=[5, 7], famp=[60, 80], chi=0.5, ndatasets=10,
                         noise=3., npts=2000)


p = Pac(idpac=(3, 1, 1), fpha=(1, 15, 1, .2), famp=(40, 100, 5, 2),
        dcomplex='wavelet', width=6)
pac = np.squeeze(p.filterfit(1024, data, data, axis=1, nperm=10)[0])

plt.figure(figsize=(12, 9))
p.comodulogram(pac.mean(-1), title=str(p), plotas='contour', ncontours=10,
               cmap='plasma', vmin=0)
plt.show()
