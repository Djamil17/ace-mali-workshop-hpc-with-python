import numpy as np
from numba import float32, jit


@jit
def f(x, y):
    return x + y


f(1, 2)


@jit(float32(float32, float32))
def f(x: float, y: float):
    return x * y


f(1.0, 2.0)


@jit(float32[:](float32[:], float32[:], float32[:]))
def f(x, y, z):
    return x + y + z


x = y = z = np.arange(5, dtype="float32")
