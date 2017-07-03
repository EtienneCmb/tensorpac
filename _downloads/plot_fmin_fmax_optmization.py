"""
==========================
Find the optimal bandwidth
==========================

Instead of looking for phase and amplitude frequency pairs (as in a
comodulogram) this example illustrate how it is possible to find starting,
ending and therefore, bandwidth coupling.
"""
from tensorpac import Pac, pac_trivec, pac_signals

sf = 256.
data, time = pac_signals(fpha=[5, 7], famp=[60, 80], noise=2, ndatasets=5,
                         npts=3000, sf=sf, dpha=10)

trif, tridx = pac_trivec(fstart=30, fend=140, fwidth=3)

p = Pac(idpac=(1, 0, 0), fpha=[5, 7], famp=trif)
pac, _ = p.filterfit(sf, data, data, axis=1)

p.triplot(pac.mean(-1), trif, tridx, cmap='Spectral_r', rmaxis=True,
          title=r'Optimal $[Fmin; Fmax]hz$ band for amplitude')

# In this example, we generated a coupling with a phase between [5, 7]hz and an
# amplitude between [60, 80]hz. To interpret the figure, the best starting
# frequency is around 50hz and the best ending frequency is around 90hz. In
# conclusion, the optimal amplitude bandwidth for this [5, 7]hz phase is
# [50, 90]hz.

# plt.savefig('triplot.png', dpi=600, bbox_inches='tight')

p.show()
