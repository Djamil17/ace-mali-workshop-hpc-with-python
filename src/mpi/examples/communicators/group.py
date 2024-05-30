"""
This program demonstrates the creation and manipulation of MPI groups and communicators
using mpi4py. It divides the processes into two groups: the first half and the second
half of the available ranks.

The key steps include:
1. Initializing the MPI environment and determining the rank and size.
2. Creating process ranks for the first half of the available ranks.
3. Creating two new groups: one including the first half and the other excluding the
first half.
4. Duplicating the first group.
5. Demonstrating various group operations such as intersection, union, and comparison.
6. Creating a communicator for the first group and checking its existence for each
process.

The program outputs the size of each group, the results of group operations, and
whether the new communicator exists for each process.
"""
from mpi4py import MPI

# 0 1 2 ... n
# n n+1 ... m
# where n == m//2


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # first row ranks to add to new group
    process_ranks = [i for i in range(size // 2)]

    group_world = comm.Get_group()

    # create new group
    first_row_group = group_world.Incl(process_ranks)
    print(f"first row size: {first_row_group.size}")

    # create new group from second row
    second_row_group = group_world.Excl(process_ranks)
    print(f"second row size: {second_row_group.size}")

    # duplicate first row group
    first_row_copy = first_row_group.Dup()

    # exhibit some methods
    print("\nrelations between groups:")
    print(f"intersect: {group_world.Intersect(first_row_group, second_row_group).size}")
    print(f"union: {group_world.Union(first_row_copy, second_row_group).size}")
    print(f"compare: {group_world.Compare(first_row_copy, second_row_group)}\n")

    first_row_comm = comm.Create(first_row_group)

    if first_row_comm != MPI.COMM_NULL:
        print(f"first_row_comm does exist in process: {rank}")
    elif first_row_comm != MPI.COMM_SELF:
        print(f"first_row_comm does not exist in process: {rank}")


if __name__ == "__main__":
    main()
