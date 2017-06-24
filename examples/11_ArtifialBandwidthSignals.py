import numpy as np
import matplotlib.pyplot as plt

from tensorpac.utils import PacSignals
from tensorpac import Pac

data, time = PacSignals(fpha=[5, 7], famp=[60, 80], chi=0.5, ndatasets=50,
                        noise=3., n=2000)


p = Pac(idpac=(1, 1, 1), fpha=(1, 15, 1, .1), famp=(40, 100, 5, 1),
        dcomplex='wavelet', width=6)
pac = np.squeeze(p.filterfit(1024, data, data, axis=1, nperm=10)[0])

p.comodulogram(pac.mean(-1), vmin=0.)
plt.show()