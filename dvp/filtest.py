import numpy as np
import matplotlib.pyplot as plt
from time import time

from tensorpac.spectral import spectral
from tensorpac.utils import PacSignals
from tensorpac.pac import Pac

from brainpipe.feature import pac


x = PacSignals(fpha=15, famp=100, ndatasets=10, tmax=1, noise=3, chi=0.8,
               dpha=0, damp=0)[0]
# x = np.squeeze(x)
print('DATASET :', x.shape)
# x = np.concatenate((x, np.random.rand(30, 1000)), axis=1)


# TENSORPAC :
t = time()
p = Pac(idpac=(4, 1, 3), fpha=(1, 30, 1, 1), famp=(60, 160, 2, 1),
        dcomplex='wavelet', filt='fir1')
xpac, spac = p.fit(1024, x, x, axis=1, nperm=100, traxis=0, njobs=-1, nblocks=10)
print('MINMAX : ', xpac.min(), xpac.max())
# print('MINMAX : ', xpac.min(), xpac.max(), spac.min(), spac.max())
# print(np.min(xpac-spac), np.max(xpac-spac))
print('Tensorpac : ', time()-t)

print('\n\n\n')
# BRAINPIPE :
t2 = time()
fpha = np.ndarray.tolist(p.fpha)
famp = np.ndarray.tolist(p.famp)
P = pac(1024, 1024, Id='513', pha_f=fpha, amp_f=famp)
XPAC = np.squeeze(P.get(x.T, x.T, n_perm=100, matricial=True, n_jobs=-1)[0])
print('Brainpipe : ', time()-t2)
print(XPAC.shape)



plt.figure(1)

plt.subplot(1, 2, 1)
plt.pcolormesh(p.xvec, p.yvec, xpac.mean(2))
plt.axis('tight')
plt.colorbar()

plt.subplot(1, 2, 2)
# plt.pcolormesh(p.xvec, p.yvec, spac.mean(2))
plt.pcolormesh(p.xvec, p.yvec, XPAC.mean(2))
plt.axis('tight')
plt.colorbar()

plt.show()