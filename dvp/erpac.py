import numpy as np
import matplotlib.pyplot as plt
from time import time

from tensorpac.utils import PacSignals
from tensorpac import Pac
from brainpipe.feature import erpac

ndatasets = 300
x1, tvec = PacSignals(fpha=10, famp=100, ndatasets=ndatasets, noise=3, npts=1000, dpha=10, damp=10)
x2 = np.random.rand(ndatasets, 700)
x = np.concatenate((x1, x2), axis=1)
T = np.arange(x.shape[1]) / 1024

print('DATASET : ', x.shape)

fpha = [8, 12]
famp = [90, 110]

t = time()
p = Pac(fpha=fpha, famp=(60, 150, 5, 1), dcomplex='wavelet', width=12)
pha = p.filter(1024, x, axis=1, ftype='phase')
amp = p.filter(1024, x, axis=1, ftype='amplitude')
print(pha.shape, amp.shape)
xpac, pval = p.erpac(pha, amp, traxis=1)
print(xpac.shape)
# 0/0
print('Tensorpac : ', time()-t)

# t = time()
# e = erpac(1024, 1500, pha_f=fpha, amp_f=famp)
# bpac = np.squeeze(e.get(x.T, x.T, n_perm=10))[0]
# print('Brainpipe : ', time()-t)
# print(bpac.shape)

# 0/0
# plt.subplot(1, 2, 1)
plt.pcolormesh(T, p.yvec, np.squeeze(xpac), cmap='Spectral_r')
plt.title('ERPAC IS IN THE HOUSE')
plt.xlabel('Time')
plt.ylabel('Amplitude')
# plt.plot(np.squeeze(xpac))
plt.axis('tight')

# plt.subplot(1, 2, 2)
# plt.plot(bpac)
# plt.axis('tight')

plt.show()