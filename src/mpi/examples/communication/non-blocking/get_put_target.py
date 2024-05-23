"""
Show the get put with target model of MPI through basic example.
Instiate a buffer in process 0 , have each process only read
a portion from window into its own part of the window.

Date: 05/22/2024
Author: Djamil Lakhdar-Hamina

"""

import numpy as np
from mpi4py import MPI
from mpi4py.util import dtlib


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    info = MPI.Info()

    datatype = MPI.FLOAT
    np_dtype = dtlib.to_numpy_dtype(datatype)
    itemsize = datatype.Get_size()

    N = size + 1
    win_size = N * itemsize if rank == 0 else 0
    win = MPI.Win.Allocate(
        size=win_size,
        disp_unit=itemsize,
        info=info,
        comm=comm,
    )

    if rank == 0:
        mem = np.frombuffer(win, dtype=np_dtype)
        print(mem)
        mem[:] = np.arange(len(mem), dtype=np_dtype)
        print(f"buffer instantiated in process 0 : {mem}")
    comm.Barrier()

    buf = np.zeros(2, dtype=np_dtype)
    target = (rank, 2, datatype)
    win.Lock(rank=0)
    win.Get(buf, target_rank=0, target=target)
    win.Unlock(rank=0)
    print(f"{rank}: {buf}")
    assert np.all(buf == [rank, rank + 1])


if __name__ == "__main__":
    main()
