from typing import Callable

import numpy as np
from mpi4py import MPI


def riemann_sum_left(
    f: Callable[[float, int], float], interval: tuple, n: int
) -> float or int:
    a = interval[0]
    b = interval[1]
    delta_x = (b - a) / n
    x = np.arange(a, b, delta_x)
    integral = sum(f(x) * delta_x)
    return integral


def f(x: float or int) -> float or int:
    return 3 * x**2


def main():
    comm = MPI.COMM_WORLD
    comm = MPI.COMM_WORLD

    root = 0
    rank = comm.Get_rank()
    size = comm.Get_size()
    global_sum = 0

    n = 1000
    interval = (1, 2)
    partition_size = (interval[1] - interval[0]) / size
    partition_number = n // size
    local_interval = (
        interval[0] + rank * partition_size,
        interval[0] + (rank + 1) * partition_size,
    )
    local_sum = riemann_sum_left(f, local_interval, partition_number)
    global_sum = comm.reduce(local_sum, op=MPI.SUM, root=root)

    if rank == 0:
        print("the global sum is:", global_sum)

    return global_sum


main()
