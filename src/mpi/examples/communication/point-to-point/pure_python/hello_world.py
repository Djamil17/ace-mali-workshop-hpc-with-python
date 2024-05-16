"""
An MPI hello world program

Date: 05/13/2024
Author : Djamil Lakhdar-Hamina

"""
from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print(f"Hello world! I am process {rank} and the size of my group size is {size}")


if __name__ == "__main__":
    main()
