"""
Demonstrate the use of reduce with numpy buffer.

Makes local array [0,1,...size] then reduces all the local arrays together.

Date: 05/09/2024
Author: Djamil Lakhdar-Hamina

"""

import numpy as np
from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    local_sum = np.arange(size, dtype=float)
    global_sum = np.empty(size, dtype=float)
    comm.Reduce([local_sum, MPI.DOUBLE], [global_sum, MPI.DOUBLE], MPI.SUM)
    if rank == 0:
        print(global_sum)


if __name__ == "__main__":
    main()
