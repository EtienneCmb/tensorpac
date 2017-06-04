import numpy as np
import matplotlib.pyplot as plt

from tensorpac.spectral import spectral
from tensorpac.utils import PacSignals
from tensorpac.pac import Pac


x = PacSignals(fpha=5, famp=120, ndatasets=30, tmax=3, noise=3, chi=0.4, dpha=0, damp=0)[0].T
# x = np.squeeze(x)
print('DATASET :', x.shape)


f = np.array([[2, 4], [20, 30], [60, 200], [30, 40], [20, 60], [7, 12]])

# xf = spectral(x, 1024, f, 1, 'amp', 'wavelet', 'bessel', 1, 6, -1)

# print(xf.shape)
p = Pac(1024, idpac=(1, 1, 1), fpha=(1, 30, 2, 2), famp=(60, 160, 10, 5), dcomplex='wavelet')
pac = p.fit(x, x, axis=0, nperm=10, traxis=1)

# 0/0


# p2 = Pac(1024, idpac=(4, 2, 0), fpha=(1, 30, 2, 2), famp=(60, 200, 10, 10), dcomplex='hilbert')
# pac2 = p2.fit(x, x, axis=0)

plt.figure(1)

# plt.subplot(1, 2, 1)
plt.pcolormesh(p.xvec, p.yvec, pac[:, :, 1])
plt.axis('tight')

# plt.subplot(1, 2, 2)
# plt.pcolormesh(p2.xvec, p2.yvec, pac2[:, :, 1])
# plt.axis('tight')

plt.show()