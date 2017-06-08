# Tensorpac
![logo](https://github.com/EtienneCmb/tensorpac/blob/master/docs/source/picture/tp.png "Hello PAC")

## Description

Tensorpac is an Python open-source toolbox for computing Phase-Amplitude Coupling (PAC) using tensors and parallel computing. On top of that, we designed a modular implementation with a relatively large amount of parameters. Checkout the [documentation](http://etiennecmb.github.io/tensorpac/) for further details.

## Installation

In a terminal, run :

```shell
git clone https://github.com/EtienneCmb/tensorpac.git tensorpac
cd tensorpac
pip setup.py install
```

## Code snippet & illustration

```python
import matplotlib.pyplot as plt
from tensorpac.utils import PacSignals
from tensorpac import Pac

# Dataset of signals artificially coupled between 10hz and 100hz :
n = 100  # number of datasets
data, time = PacSignals(fpha=10, famp=100, noise=3, ndatasets=n, dpha=10, damp=10)

# Extract PAC :
p = Pac(idpac=(4, 0, 0), fpha=(2, 30, 1, 1), famp=(60, 150, 5, 5),
        dcomplex='wavelet', width=12)
xpac, pval = p.filterfit(1024, data, data, axis=1, nperm=210)

# Plot your Phase-Amplitude Coupling :
p.comodulogram(xpac.mean(-1), title='Contour plot with 5 regions',
               cmap='Spectral_r', plotas='contour', ncontours=5, vmin=60, vmax=300)

plt.show()
```
![logo](https://github.com/EtienneCmb/tensorpac/blob/master/docs/source/picture/readme.png "Comodulogram")

## Contributors

- [Etienne Combrisson](http://etiennecmb.github.io)
- Juan L.P. Soto
- [Karim Jerbi](www.karimjerbi.com)

