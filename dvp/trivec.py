from tensorpac import Pac
from tensorpac.utils import PacTriVec, PacSignals
import matplotlib.pyplot as plt

data, time = PacSignals(famp=120, noise=3, tmax=1, ndatasets=1, dpha=10, damp=10)

ampvec, ampidx = PacTriVec(fwidth=2, fend=200)  # 
p = Pac(idpac=(2, 4, 1), fpha=[9, 11], famp=ampvec,
        dcomplex='hilbert', width=12)


xpac = p.filterfit(1024, data, data, axis=1)[0]
plt.subplot(1, 2, 1)
p.triplot(xpac, ampvec, ampidx, plotas='imshow', vmin=0.,
          title='Complex decomposition : Hilbert\n' + p.method)

p.dcomplex = 'wavelet'
xpac = p.filterfit(1024, data, data, axis=1)[0]
plt.subplot(1, 2, 2)
p.triplot(xpac, ampvec, ampidx, plotas='imshow', vmin=0.,
          title='Complex decomposition : Wavelet\n' + p.method)

plt.show()
