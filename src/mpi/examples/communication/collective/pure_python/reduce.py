"""
Shows the use of reduce. Each process produces
a random integer and then all are summed and sent to
process 0 (root).

Date: 05/16/2024
Author: Djamil Lakhdar-Hamina

"""

from random import randint

from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    LOWER_LIMIT = 1
    UPPER_LIMIT = 10

    randomn_int = randint(LOWER_LIMIT, UPPER_LIMIT)
    print(f"rank {rank}: {randomn_int}")
    global_sum = comm.reduce(randomn_int, op=MPI.SUM, root=0)

    if rank == 0:
        print("the global sum is:", global_sum)


if __name__ == "__main__":
    main()
