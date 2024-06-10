"""



"""
import time
from functools import wraps

import numba as nb
import numpy as np


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} Took {total_time:.4f} s")
        return result

    return timeit_wrapper


@timeit
def no_jit_bubble_sort(x: np.ndarray, in_place=False, out=None) -> np.ndarray:
    n = x.shape[0]
    for i in range(n):
        for j in range(i, n):
            if x[j - 1] > x[j]:
                swap = x[j - 1]
                x[j] = x[j - 1]
                x[j] = swap
    return x


@timeit
@nb.jit
def jit_bubble_sort(x: np.ndarray, in_place=False, out=None) -> np.ndarray:
    n = x.shape[0]
    for i in range(n):
        for j in range(i, n):
            if x[j - 1] > x[j]:
                swap = x[j - 1]
                x[j] = x[j - 1]
                x[j] = swap
    return x


def main() -> None:
    rng = np.random.RandomState(42)
    x = rng.randint(1, 10, size=1000)
    x_ = x.copy()
    jit_bubble_sort(x_)
    no_jit_bubble_sort(x)


if __name__ == "__main__":
    main()
