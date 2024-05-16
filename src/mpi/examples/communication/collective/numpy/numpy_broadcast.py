import numpy as np
from mpi4py import MPI


def main():
    ROOT = 0
    m = 1
    n = 3
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Broadcast
    if rank == 0:
        sendbuf = np.random.rand(m, n)
    else:
        # Cannot be none, the sendbuf across processes must be of same type
        sendbuf = np.zeros((m, n), dtype=float)

    comm.Bcast([sendbuf, MPI.FLOAT], root=ROOT)

    if rank != 0:
        print(f"received broadcasted {sendbuf} at {rank}")


main()
