import numpy as np
from scipy.stats import pearsonr
from scipy.stats import chi2

a = np.random.rand(100).reshape(20, 5)
b = np.random.rand(150).reshape(30, 5)

p = np.zeros((20, 30))
for k in range(20):
    for i in range(30):
        p[k, i] = pearsonr(a[k, ...], b[i, ...])[0]

def generate_correlation_map(x, y):
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must ' +
                         'have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x, y.T) - n * np.dot(mu_x[:, np.newaxis], mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])

def pear(x, y, axis=-1):
    if axis not in [-1, x.ndim]:
        x = np.swapaxes(x, -1, axis)
        y = np.swapaxes(y, -1, axis)
    mu_x, mu_y = x.mean(-1, keepdims=True), y.mean(-1, keepdims=True)
    n = x.shape[-1]
    s_x, s_y = x.std(1, ddof=n-1, keepdims=True), y.std(1, ddof=n-1, keepdims=True)
    xy = np.einsum('i...j, k...j->ik...', x, y)
    mu_xy = np.einsum('i...j, k...j->ik...', mu_x, mu_y)
    cov = xy - n * mu_xy
    return cov / np.einsum('i...j, k...j->ik...', s_x, s_y)

# p2 = generate_correlation_map(a, b)
p3 = pear(a, b)

# print(p.shape, p2.shape, p3.shape)
# print(np.allclose(p2, p3))

def circ_corrcc(alpha, x):
    n = alpha.shape[-1]
    # Compute correlation coefficent for sin and cos independently
    sa, ca = np.sin(alpha), np.cos(alpha)
    rxs = pear(x, sa)[0]
    rxc = pear(x, ca)[0]
    rcs = pear(sa, ca)[0]

    # Compute angular-linear correlation (equ. 27.47)
    rho = np.sqrt((rxc**2 + rxs**2 - 2*rxc*rxs*rcs)/(1-rcs**2))

    # Compute pvalue
    pval = 1 - chi2.cdf(n*rho**2, 2)

    return rho, pval

# circ_corrcc(a, b)
