"""
Show use of allgather, have each process record its
rank in a variable then gather all these variables in list.

Date: 05/16/2024
Author: Djamil Lakhdar-Hamina

"""
from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    ranks = comm.allgather(rank)

    print(f"{ranks} on process {rank}")


if __name__ == "__main__":
    main()
