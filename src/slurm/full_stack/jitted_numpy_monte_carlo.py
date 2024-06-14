import sys

import numba as nb
import numpy as np
from mpi4py import MPI


@nb.jit(nb.float32(nb.int32, nb.float32), fastmath=True, parallel=True)
def parallel_jitted_monte_carlo_pi(samples: nb.int32, area: nb.float32) -> nb.float32:
    hit = 0.0
    for _ in nb.prange(samples):
        x, y = np.random.rand(), np.random.rand()
        if (x**2 + y**2) ** (1 / 2) <= 1.0:
            hit += 1.0

    return area * hit / samples


def main() -> None:
    AREA = 4.0

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = int(sys.argv[1])
    n = N // size

    local_pi = parallel_jitted_monte_carlo_pi(samples=n, area=AREA)
    pi = comm.reduce(local_pi, op=MPI.SUM)
    if rank == 0:
        return pi / size


if __name__ == "__main__":
    average_pi = main()
