import numpy as np
import matplotlib.pyplot as plt
from time import time

from tensorpac.spectral import spectral
from tensorpac.utils import PacSignals
from tensorpac.pac import Pac

from brainpipe.feature import pac


x = PacSignals(fpha=15, famp=100, ndatasets=100, tmax=1, noise=1, chi=0.,
               dpha=0, damp=0)[0]
# x = np.squeeze(x)
x = np.concatenate((x, np.random.rand(100, 1000)), axis=1)
print('DATASET :', x.shape)


# TENSORPAC :
t = time()
p = Pac(idpac=(5, 0, 0), fpha=[12, 17], famp=(60, 160, 2, 1),
        dcomplex='hilbert', filt='fir1')
xpac, spac = p.fit(1024, x, x, axis=1, nperm=100, traxis=0, njobs=-1, nblocks=10)
print('MINMAX : ', xpac.min(), xpac.max())
# print('MINMAX : ', xpac.min(), xpac.max(), spac.min(), spac.max())
# print(np.min(xpac-spac), np.max(xpac-spac))
print('Tensorpac : ', time()-t)

print('\n\n\n')
# BRAINPIPE :
# t2 = time()
# fpha = np.ndarray.tolist(p.fpha)
# famp = np.ndarray.tolist(p.famp)
# P = pac(1024, 1024, Id='513', pha_f=fpha, amp_f=famp)
# XPAC = np.squeeze(P.get(x.T, x.T, n_perm=100, matricial=True, n_jobs=-1)[0])
# print('Brainpipe : ', time()-t2)
# print(XPAC.shape)



# plt.figure(1)

# plt.subplot(1, 2, 1)
# plt.pcolormesh(xpac[30, ...])
# # plt.pcolormesh(p.xvec, p.yvec, xpac[30, ...])
# plt.axis('tight')
# plt.colorbar()

# # plt.subplot(1, 2, 2)
# # # plt.pcolormesh(p.xvec, p.yvec, spac.mean(2))
# # plt.pcolormesh(p.xvec, p.yvec, XPAC.mean(2))
# # plt.axis('tight')
# # plt.colorbar()

# plt.show()



# plt.pcolormesh(p.xvec, p.yvec, xpac[:, :, 0:1024].mean(2))
plt.pcolormesh(np.arange(x.shape[1]), p.yvec, xpac[:, 0, :])
# plt.clim(0, 1)
plt.axis('tight')
plt.colorbar()
plt.show()