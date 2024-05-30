"""
This program demonstrates the use of MPI to split a communicator into two
sub-communicators based on the rank of the processes. It initializes MPI,
determines the rank and size of the world communicator, and then divides
the processes into two groups.

Processes with ranks in the first half of the communicator are assigned to one group,
while those in the second half are assigned to another. Each group performs
communication within its sub-communicator and prints out its sub-rank and the size
of the sub-communicator.

The key steps include:
1. Initializing the MPI environment and obtaining the rank and size.
2. Splitting the communicator into two sub-communicators based on rank.
3. Performing communication within each sub-communicator.
4. Printing the sub-rank and sub-communicator size for each process.
5. Freeing the sub-communicator resources.

Date: 05/30/2024
Author: Djamil Lakhdar-Hamina
"""
from mpi4py import MPI


def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    assert size > 2

    # Split the communicator into two sub-communicators
    if rank < size // 2:
        color = 0
        key = rank
    else:
        color = 1
        key = rank

    # put into one of two groups if data defined in one or the other is 0 or 1
    sub_comm = comm.Split(color, key)

    # Perform communication within the sub-communicator
    sub_rank = sub_comm.Get_rank()
    sub_size = sub_comm.Get_size()

    print(f"Rank {rank}: Sub-rank {sub_rank} of size {sub_size}")

    sub_comm.Free()


if __name__ == "__main__":
    main()
