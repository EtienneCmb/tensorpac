import numpy as np
import matplotlib.pyplot as plt
from time import time

from tensorpac.utils import PacSignals
from tensorpac import Pac
from brainpipe.feature import erpac

ndatasets = 300
x1, tvec = PacSignals(fpha=10, famp=100, ndatasets=ndatasets, noise=1, tmax=2)
x2, tvec = PacSignals(fpha=20, famp=150, ndatasets=ndatasets, noise=1, tmax=2)
x = np.concatenate((x1, x2), axis=1) + 5.

print('DATASET : ', x.shape)

fpha = [9, 11]
famp = [90, 110]

t = time()
p = Pac(idpac=(6, 0, 0), fpha=fpha, famp=famp, dcomplex='hilbert', width=12, nblocks=10)
xpac, spac = p.fit(x, x, axis=1, nperm=10, traxis=0, njobs=-1)
print(xpac.shape)
# 0/0
print('Tensorpac : ', time()-t)

# t = time()
# e = erpac(1024, 2048, pha_f=fpha, amp_f=famp)
# bpac = np.squeeze(e.get(x.T, x.T, n_perm=10))[0]
# print('Brainpipe : ', time()-t)
# print(bpac.shape)

# 0/0
plt.subplot(1, 2, 1)
plt.plot(np.squeeze(xpac))
plt.axis('tight')

# plt.subplot(1, 2, 2)
# plt.plot(bpac)
# plt.axis('tight')

plt.show()