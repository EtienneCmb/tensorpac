from spectrum import bicoherence
from tensorpac import pac_signals_wavelet
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab

y, _ = pac_signals_wavelet(ntrials=1)
y = np.ravel(y)
# bicoherence(y)

yfft = np.fft.fft(y)
nf = np.fft.fftfreq(len(y), 1/1024)

f1, f2 = 10, 100

# print(nf[0:2000])

# pylab.figure()
# pylab.subplot(121)
# pylab.plot( nf, np.abs(yfft) )
# pylab.subplot(122)
# pylab.plot(nf, np.angle(yfft) )
# pylab.show()

# def compute_bicoherence(s1, s2, rate, nperseg=1024, noverlap=512):
#     """ Compute the bicoherence between two signals of the same lengths s1 and s2
#     using the function scipy.signal.spectrogram
#     """
#     # compute the stft
#     f1, t1, spec_s1 = signal.spectrogram(s1, fs = rate, nperseg = nperseg, noverlap = noverlap, mode = 'complex',)
#     f2, t2, spec_s2 = signal.spectrogram(s2, fs = rate, nperseg = nperseg, noverlap = noverlap, mode = 'complex')
#     print(spec_s1.shape)
#     # transpose (f, t) -> (t, f)
#     spec_s1 = np.transpose(spec_s1, [1, 0])
#     spec_s2 = np.transpose(spec_s2, [1, 0])

#     # compute the bicoherence
#     arg = np.arange(f1.size / 2)
#     sumarg = arg[:, None] + arg[None, :]
#     print(arg, sumarg)
#     num = np.abs(
#         np.mean(spec_s1[:, arg, None] * spec_s1[:, None, arg] * np.conjugate(spec_s2[:, sumarg]), 
#         axis = 0)
#         ) ** 2
#     denum = np.mean(
#         np.abs(spec_s1[:, arg, None] * spec_s1[:, None, arg]) ** 2, axis = 0) * np.mean(
#             np.abs(np.conjugate(spec_s2[:, sumarg])) ** 2, 
#             axis = 0)
#     bicoh = num / denum
#     return f1[arg], bicoh

# # exemple of use and display
# freqs, bicoh = compute_bicoherence(y, y, 1024, nperseg=256, noverlap=128)
# f = plt.figure(figsize = (9, 9))
# plt.pcolormesh(freqs, freqs, bicoh, 
#     # cmap = 'inferno'
#     )
# plt.colorbar()
# plt.clim(0, 0.5)
# plt.show()