import numba as nb
import numpy as np


@nb.jit(nopython=True, nogil=True, nocache=False, parallel=True)
def f(x, y):
    return x + y


f(1, 2)


@nb.jit(nb.float32(nb.float32, nb.float32))
def f(x: float, y: float):
    return x * y


f(1.0, 2.0)


@nb.jit(nb.float32[:](nb.float32[:], nb.float32[:], nb.float32[:]))
def f(x, y, z):
    return x + y + z


x = y = z = np.arange(5, dtype="float32")
x
f(x, y, z)
