"""
This program estimates the value of π (pi) using the Monte Carlo method with
different levels of optimization and precision. It uses the Numba library to
optimize the calculations and includes a decorator to measure the execution time
of each function.

The main steps are:
1. Importing necessary libraries (`numpy` for array handling, `numba` for
   performance optimization, `functools` for wrapping functions, and `time` for
   measuring execution time).
2. Defining a `timeit` decorator to measure and return the execution time of
   functions.
3. Implementing several versions of the Monte Carlo π estimation function:
   - `monte_carlo_pi`: A plain NumPy implementation.
   - `jitted_monte_carlo_pi`: A Numba-jitted implementation.
   - `less_precision_jitted_monte_carlo_pi`: A Numba-jitted implementation with
     less precision.
   - `fast_math_jitted_monte_carlo_pi`: A Numba-jitted implementation with
     `fastmath` enabled.
   - `parallel_jitted_monte_carlo_pi`: A Numba-jitted implementation with
     `fastmath` and parallel computation enabled.
4. Using the `main` function to execute and compare the performance and
   precision of these implementations.

Usage:
------
Simply run the script to see the output of the estimated values of π and the
execution times for each implementation.

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
        return result, total_time

    return timeit_wrapper


@timeit
def monte_carlo_pi(samples: (int | float), area) -> int | float:
    hit = 0
    samples_ = np.copy(samples)
    while samples_ > 0:
        samples_ = samples_ - 1
        x, y = np.random.rand(), np.random.rand()
        if (x**2 + y**2) ** (1 / 2) <= 1:
            hit += 1
    return area * hit / samples


@timeit
@nb.jit(nb.float64(nb.int64, nb.float64))
def jitted_monte_carlo_pi(samples: nb.int32, area: nb.float32) -> nb.float64:
    hit = 0.0
    for _ in nb.prange(samples):
        x, y = np.random.rand(), np.random.rand()
        if (x**2 + y**2) ** (1 / 2) <= 1.0:
            hit += 1.0

    return area * hit / samples


@timeit
@nb.jit(nb.float32(nb.int32, nb.float32))
def less_precision_jitted_monte_carlo_pi(
    samples: nb.int32, area: nb.float32
) -> nb.float32:
    hit = 0.0
    for _ in nb.prange(samples):
        x, y = np.random.rand(), np.random.rand()
        if (x**2 + y**2) ** (1 / 2) <= 1.0:
            hit += 1.0

    return area * hit / samples


@timeit
@nb.jit(nb.float32(nb.int32, nb.float32), fastmath=True)
def fast_math_jitted_monte_carlo_pi(samples: nb.int32, area: nb.float32) -> nb.float32:
    hit = 0.0
    for _ in nb.prange(samples):
        x, y = np.random.rand(), np.random.rand()
        if (x**2 + y**2) ** (1 / 2) <= 1.0:
            hit += 1.0

    return area * hit / samples


@timeit
@nb.jit(nb.float32(nb.int32, nb.float32), fastmath=True, parallel=True)
def parallel_jitted_monte_carlo_pi(samples: nb.int32, area: nb.float32) -> nb.float32:
    hit = 0.0
    for _ in nb.prange(samples):
        x, y = np.random.rand(), np.random.rand()
        if (x**2 + y**2) ** (1 / 2) <= 1.0:
            hit += 1.0

    return area * hit / samples


def main() -> str:
    samples = 1e7
    area = 4.0
    num_execs = 25
    functions = [
        monte_carlo_pi,
        jitted_monte_carlo_pi,
        less_precision_jitted_monte_carlo_pi,
        fast_math_jitted_monte_carlo_pi,
        parallel_jitted_monte_carlo_pi,
    ]

    for func in functions:
        times, pis = [], []
        for _ in range(num_execs):
            pi, time = func(samples=samples, area=area)
            pis.append(pi)
            times.append(time)
        ave_pi, ave_time = np.mean(np.array([pis, times]), axis=1)
        print(f"{func.__name__}\npi ≈ {ave_pi}\ntime:{ave_time}\n")


if __name__ == "__main__":
    main()
