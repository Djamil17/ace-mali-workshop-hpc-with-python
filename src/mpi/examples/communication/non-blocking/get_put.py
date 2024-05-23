"""
Show use of get put model. Program instantiates a window of
shared memory which only exists in process 0. Process 0
fills a buffer, or puts data, and then the other processes
get that data from shared memory.

Date: 05/22/2024
Author: Djamil Lakhdar-Hamina

"""

import numpy as np
from mpi4py import MPI
from mpi4py.util import dtlib


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    dtype = MPI.FLOAT
    np_dtype = dtlib.to_numpy_dtype(dtype)
    itemsize = dtype.Get_size()

    N = 42
    win_size = N * itemsize if rank == 0 else 0
    win = MPI.Win.Allocate(win_size, comm=comm)
    buf = np.empty(N, dtype=np_dtype)

    if rank == 0:
        buf.fill(42)
        win.Lock(rank=0, lock_type=MPI.LOCK_EXCLUSIVE, assertion=MPI.MODE_NOCHECK)
        win.Put(buf, target_rank=0)
        win.Unlock(rank=0)
        comm.Barrier()
    else:
        comm.Barrier()
        win.Lock(rank=0, lock_type=MPI.LOCK_SHARED, assertion=MPI.MODE_NOCHECK)
        win.Get(buf, target_rank=0)
        win.Unlock(rank=0)
        assert np.all(buf == 42)


if __name__ == "__main__":
    main()
