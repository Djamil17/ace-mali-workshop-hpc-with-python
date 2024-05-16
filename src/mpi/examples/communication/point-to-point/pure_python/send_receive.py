"""
Have each worker node produce a random integer, then send it to the root process
to be printed by root.

Date: 05/13/2024
Author : Djamil Lakhdar-Hamina

"""

from random import randint

from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    LOWER_LIMIT = 1
    UPPER_LIMIT = 10

    if rank != 0:
        data = randint(LOWER_LIMIT, UPPER_LIMIT)
        comm.send(data, dest=0)
    else:
        for r in range(1, size):
            data = comm.recv(source=r)
            print(f"my name is rank 0 and I received the number {data} from rank {r}")


if __name__ == "__main__":
    main()
