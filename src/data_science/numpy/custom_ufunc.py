import time
from functools import wraps

import numpy as np


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__}{args}{kwargs} Took {total_time:.4f} s")
        return result

    return timeit_wrapper


def mag(x: (int | float), y: (int | float)) -> int | float:
    return (x**2 + y**2) ** (1 / 2)


docstring = """"
Calculate the magntiude between two elements element by element in array

    This function two arrays, finds the magnitude of 1rst element and 1rst
    element of second array , 2nd element and 2nd element of the second array
    etc.

    Parameters:
    x (np.ndarray[(int | float)]): The first array
    y (np.ndarray[(int | float)]): The second array

    Returns:
    np.ndarray: the array with each magnitude calculated


"""

dtype = np.float32
mag_vect = np.vectorize(mag, doc=docstring, otype=dtype)
x = y = np.arange(10)
x = mag_vect(x, 1)
x
x = mag_vect(x, y)
x
type(x[0])

mag_vect = np.vectorize(mag, doc=docstring, cache=True, signature="(m),(m) -> (m)")
x = mag_vect(x, 1)
x = mag_vect(x, y)
x


def polyval(p, x):
    _p = list(p)
    res = 0
    while _p:
        co = _p.pop(0)
        res += co**x
    return res


vpolyval = np.vectorize(polyval, excluded=["p"])
vpolyval(p=[1, 2, 3], x=[0, 1])


@np.vectorize(doc=docstring, cache=True, signature="(m),(m) -> (m)")
def mag(x: (int | float), y: (int | float)) -> int | float:
    return (x**2 + y**2) ** (1 / 2)
