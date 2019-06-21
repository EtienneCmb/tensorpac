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

from tensorpac import Pac, pac_signals_tort

data, time = pac_signals_tort(f_pha=[5, 7], f_amp=[60, 80], chi=0.5,
                              n_epochs=10, noise=3., n_times=2000)


p = Pac(idpac=(3, 1, 1), f_pha=(1, 15, 1, .2), f_amp=(40, 100, 5, 2),
        dcomplex='wavelet', width=6)
pac = p.filterfit(1024, data, n_perm=10)

plt.figure(figsize=(12, 9))
p.comodulogram(pac.mean(-1), title=str(p), plotas='contour', ncontours=10,
               cmap='plasma', vmin=0)
plt.show()
