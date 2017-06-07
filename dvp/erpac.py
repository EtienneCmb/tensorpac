import numpy as np
import matplotlib.pyplot as plt
from time import time

from tensorpac.utils import PacSignals
from tensorpac import Pac
from brainpipe.feature import erpac

ndatasets = 300
x1, tvec = PacSignals(fpha=10, famp=100, ndatasets=ndatasets)
x2, tvec = PacSignals(fpha=20, famp=150, ndatasets=ndatasets)
x = np.concatenate((x1, x2), axis=1)

print('DATASET : ', x.shape)

fpha = [8, 15]
famp = [80, 120]

t = time()
p = Pac(idpac=(5, 0, 0), fpha=fpha, famp=famp, dcomplex='hilbert')
xpac, spac = p.fit(1024, x, x, axis=1, nperm=10, traxis=0, njobs=-1, nblocks=10)
print('Tensorpac : ', time()-t)

t = time()
e = erpac(1024, 2048, pha_f=fpha, amp_f=famp)
bpac = np.squeeze(e.get(x.T, x.T, n_perm=100))[0]
print('Brainpipe : ', time()-t)
print(bpac.shape)


plt.subplot(1, 2, 1)
plt.plot(np.squeeze(xpac))
plt.axis('tight')

plt.subplot(1, 2, 2)
plt.plot(bpac)
plt.axis('tight')

plt.show()