"""
This program demonstrates basic MPI (Message Passing Interface) asynchronous
communication usingthe mpi4py library. The example involves two processes
(ranks 0 and 1) exchanging messages.

Usage:
    Run the script with an MPI execution command such as
   `mpiexec -n 2 python script_name.py`.

Date: 05/23/2024
Author: Djamil Lakhdar-Hamina

"""

import numpy as np
from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    status = MPI.Status()

    assert size == 2
    N = 10
    # non numpy-example
    if rank == 0:
        data = bytearray("Hello, world!".encode())
        request = comm.Isend(data, dest=1)
        print(2 + 3)
    else:
        buf = bytearray(46)
        request = comm.Irecv(buf, source=0)
        print(6 + 4)
        request.wait(status=status)
        print(f"{buf.decode()} From process {rank}")

    # numpy-example
    if rank == 0:
        data = np.random.rand(N)
        request = comm.Isend([data, MPI.FLOAT], dest=1)
        print(2 + 3)
    else:
        buf = np.empty(N)
        request = comm.Irecv([buf, MPI.FLOAT], source=0)
        print(3 + 3)
        request.wait(status=status)
        print(f"f process {rank}: {buf}")


if __name__ == "__main__":
    main()
