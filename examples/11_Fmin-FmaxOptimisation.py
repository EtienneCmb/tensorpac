"""Search for the optimal [Fmin; Fmax]hz amplitude band."""
import matplotlib.pyplot as plt

from tensorpac import Pac, pac_trivec, pac_signals

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sf = 256.
data, time = pac_signals(fpha=[5, 7], famp=[60, 100], noise=2, ndatasets=30,
                         npts=6000, sf=sf, dpha=10)

trif, tridx = pac_trivec(fstart=30, fend=160, fwidth=5)

p = Pac(idpac=(1, 0, 0), fpha=[5, 7], famp=trif)
pac, _ = p.filterfit(sf, data, data, axis=1)

p.triplot(pac.mean(-1), trif, tridx, cmap='Spectral_r', rmaxis=True,
          title=r'Optimal $[Fmin; Fmax]$hz band for amplitude')
# plt.savefig('triplot.png', dpi=600, bbox_inches='tight')
p.show()
