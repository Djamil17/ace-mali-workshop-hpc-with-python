"""
Have each node produce a random integer, then the two nodes exchange
their random integers.

Date: 05/13/2024
Author : Djamil Lakhdar-Hamina

"""

from random import randint

from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    assert size == 2

    LOWER_LIMIT = 1
    UPPER_LIMIT = 10

    if rank == 0:
        local_int = randint(LOWER_LIMIT, UPPER_LIMIT)
        new_local_int = comm.sendrecv(sendobj=local_int, dest=1, source=MPI.ANY_SOURCE)
        print(
            f"my name is rank 0 and I received the number {new_local_int}\
from rank {rank}"
        )
    else:
        local_int = randint(LOWER_LIMIT, UPPER_LIMIT)
        new_local_int = comm.sendrecv(sendobj=local_int, dest=0, source=MPI.ANY_SOURCE)
        print(
            f"my name is rank 0 and I received the number {new_local_int}\
from rank {rank}"
        )


if __name__ == "__main__":
    main()
