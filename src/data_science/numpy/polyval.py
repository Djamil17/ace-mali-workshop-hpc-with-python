import numpy as np


def polyval(p, x):
    _p = list(p)
    res = 0
    while _p:
        co = _p.pop(0)
        res += co**x
    return res


polyval = np.vectorize(polyval)
print(polyval([1, 2, 3], 1))


@np.vectorize(polyval)
def polyval(p, x):
    _p = list(p)
    res = 0
    while _p:
        co = _p.pop(0)
        res += co**x
    return res


vpolyval = np.vectorize(polyval, excluded=["p"])
print(vpolyval(p=[1, 2, 3], x=[0, 1]))
