"""
=====================================
Compute the ERPAC (Voytek et al 2013)
=====================================

Event-Related Phase-Amplitude Coupling (ERPAC) do not measure PAC across time
cycle but instead, across trials (just as proposed JP. Lachaux with the
PLV/PLS). Measuring across trials enable to have a real-time estimation of PAC.
Warning, depending on your data, even with tensor calculation the ERPAC is
significantly slower. Don't worry, take a coffee.

In this example, we generate a signal that have a 10<->100hz coupling the first
1000 points, then, the 700 following points are noise.
"""
import numpy as np
from tensorpac import Pac, pac_signals_tort

# Generate a 10<->100hz coupling :
n_trials = 300
n_pts = 1000
sf = 1024.
x1, tvec = pac_signals_tort(f_pha=10, f_amp=100, n_trials=n_trials, noise=2,
                            n_pts=n_pts, dpha=10, damp=10, sf=sf)

# Generate noise and concatenate the coupling and the noise :
x2 = np.random.rand(n_trials, 700)
x = np.concatenate((x1.squeeze(), x2), axis=1)  # Shape : (n_trials, n_pts)
time = np.arange(x.shape[1]) / sf
x = x[:, np.newaxis, :]

# Define a PAC object :
p = Pac(f_pha=[9, 11], f_amp=(60, 140, 5, 1))

# Extract the phase and the amplitude :
pha = p.filter(sf, x, ftype='phase')      # Shape (npha, n_trials, 1, n_pts)
amp = p.filter(sf, x, ftype='amplitude')  # Shape (namp, n_trials, 1, n_pts)

# Compute the ERPAC and use the traxis to specify that the trial axis is the
# first one :
erpac = p.erpac(pha, amp, method='gc')
pval = p.pvalues_

# Remove unused dimensions :
erpac, pval = np.squeeze(erpac), np.squeeze(pval)

# Plot without p-values :
p.pacplot(erpac, time, p.yvec, xlabel='Time (second)', cmap='Spectral_r',
          ylabel='Amplitude frequency', title=str(p), cblabel='ERPAC',
          vmin=0., rmaxis=True)

# Plot with every non-significant values masked in gray :
# p.pacplot(erpac, time, p.yvec, xlabel='Time (second)', cmap='Spectral_r',
#           ylabel='Amplitude frequency', title='ERPAC example', vmin=0.,
#           vmax=1., pvalues=pval, bad='lightgray', plotas='contour',
#           cblabel='ERPAC')

# Plot with significiendy levels :
# p.pacplot(erpac, time, p.yvec, xlabel='Time (second)', cmap='Spectral_r',
#           ylabel='Amplitude frequency', title='ERPAC example', vmin=0.,
#           vmax=1., pvalues=pval, levels=[1e-20, 1e-10, 1e-2, 0.05],
#           levelcmap='inferno', plotas='contour', cblabel='ERPAC')
# p.savefig('erpac.png', dpi=300)
p.show()
