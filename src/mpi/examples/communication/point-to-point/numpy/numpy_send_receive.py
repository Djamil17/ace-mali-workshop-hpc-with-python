"""
Exhibit the use of memory-buffer from np in mpi4py.

Date: 05/22/2024
Author: Djamil Lakhdar-Hamina

"""

import numpy as np
from mpi4py import MPI


def main():
    ROOT = 0
    m = 1
    n = 3
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Send , Receive
    if rank == 0:
        for p in range(1, size):
            sendbuf = np.random.rand(m, n)
            comm.Send([sendbuf, MPI.FLOAT], dest=p)
            print(f"sending {sendbuf} to {p} from root")

    else:
        rcvbuf = np.zeros((m, n), dtype=float)
        comm.Recv([rcvbuf, MPI.FLOAT], source=ROOT)
        print(f"received {rcvbuf} at {rank}")


if __name__ == "__main__":
    main()
