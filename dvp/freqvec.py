import numpy as np
from tensorpac.utils import _CheckFreq

f = np.array([[1, 2], [4, 5], [6, 7]]).T
print(f.shape)


print(_CheckFreq(f).shape)