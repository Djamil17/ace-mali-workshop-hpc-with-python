from typing import Callable

import numpy as np
from mpi4py import MPI
from mpi4py.util import dtlib


def riemann_sum_left(
    f: Callable[[float, int], float], interval: tuple, n: int
) -> float or int:
    delta_x = (interval[1] - interval[0]) / n
    x = [interval[0] + i * delta_x for i in range(0, int(n))]
    sum = 0
    for i in range(0, int(n)):
        sum += f(x[i]) * delta_x
    return sum


def f(x):
    return 2 * x


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    dtype = MPI.FLOAT
    np_dtype = dtlib.to_numpy_dtype(dtype)
    itemsize = dtype.Get_size()

    n = 1000
    interval = (0, 2)
    partition_size = (interval[1] - interval[0]) / size
    local_interval = (
        interval[0] + rank * partition_size,
        interval[0] + (rank + 1) * partition_size,
    )

    local_sum = np.array([riemann_sum_left(f, local_interval, n)], dtype=np_dtype)
    global_sum = np.empty(1, dtype=np_dtype)
    win = MPI.Win.Allocate(itemsize, comm=comm)
    win.Lock(rank=0)
    target_rank = 0
    win.Accumulate([local_sum, dtype], target_rank=target_rank, op=MPI.SUM)
    win.Unlock(rank=0)
    comm.Barrier()
    win.Lock(rank=0)
    win.Get(global_sum, target_rank=0)
    win.Unlock(rank=0)

    if rank == 0:
        print(f"The integral of 2x from {interval[0]}-{interval[1]} = {global_sum[0]}")

    win.Free()


main()
