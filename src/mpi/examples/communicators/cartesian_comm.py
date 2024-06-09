"""
This program demonstrates the use of MPI (Message Passing Interface) to create
a 2D Cartesian grid topology with periodic boundaries using the mpi4py library.

The program checks if the number of processes (MPI ranks) is a perfect square,
ensuring it can form a square grid. Each process determines its coordinates
within this grid and identifies its neighboring processes, facilitating
communication between them.

The program involves the following key steps:
1. Initialize the MPI environment and obtain the rank and size of the communicator.
2. Ensure the number of processes is a perfect square, allowing for an n x n grid.
3. Create a Cartesian communicator with periodic boundaries.
4. Print the coordinates of each rank within the grid.
5. Perform shift communications to identify left-right and up-down neighbors.
6. Create and use a sub-communicator for communication along a specified dimension.
7. Clean up by freeing the Cartesian communicators.

The program showcases essential MPI functionalities for parallel computing
applications, particularly in simulations and computations requiring structured
grid topologies.

Date: 05/30/2024
Author: Djamil Lakhdar-Hamina
"""
from mpi4py import MPI


def check_perfect_square(x: int) -> bool:
    """
    Check if a given integer is a perfect square.

    This function takes an integer as input and determines if it is a perfect
    square. A perfect square is an integer that is the square of another integer.

    Parameters:
    x (int): The integer to be checked.

    Returns:
    bool: True if x is a perfect square, False otherwise.

    Example:
    >>> check_perfect_square(16)
    True
    >>> check_perfect_square(18)
    False
    """
    square = x ** (1 / 2)
    if square % 1 == 0:
        return True
    else:
        return False


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    assert size >= 4 and check_perfect_square(size)
    # want to create a nxn square grid with periodicity
    n = size // 2
    cart_comm = comm.Create_cart(dims=[n, n], periods=[True, True], reorder=True)

    print(f"rank {rank} has coordinates {cart_comm.Get_coords(rank)}")

    comm.Barrier()
    # Perform communication within the Cartesian communicator
    # (e.g., shift communication)
    left_neighbor, right_neighbor = cart_comm.Shift(1, 1)
    print(
        f"Rank {rank}: Left neighbor {left_neighbor}, Right neighbor {right_neighbor}"
    )

    up_neighbor, down_neighbor = cart_comm.Shift(0, 1)
    print(f"Rank {rank}: Up neighbor {up_neighbor}, Down neighbor {down_neighbor}")

    comm.Barrier()
    sub_cart_comm = cart_comm.Sub([False, True])

    print(f"Rank {rank}: has sub-coordinates {sub_cart_comm.coords}")

    # Free the Cartesian communicators
    sub_cart_comm.Free()
    cart_comm.Free()


if __name__ == "__main__":
    main()
