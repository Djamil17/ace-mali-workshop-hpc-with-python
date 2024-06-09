import numpy as np
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
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Define the global size of the array
    global_array_size = 100

    # Create the global array on rank 0
    if rank == 0:
        global_array = np.arange(global_array_size, dtype="i")
    else:
        global_array = None

    # Calculate local array size for block distribution
    local_array_size = global_array_size // size

    # Define the arguments for Create_darray
    gsizes = [global_array_size]  # Global size of the array
    distribs = [MPI.DISTRIBUTE_BLOCK]  # Block distribution
    dargs = [MPI.DISTRIBUTE_DFLT_DARG]  # Default argument for block size
    psizes = [size]  # Number of processes

    # Create the distributed array datatype
    dtype = MPI.FLOAT

    darray_type = dtype.Create_darray(
        size=size,
        rank=rank,
        gsizes=gsizes,
        distribs=distribs,
        dargs=dargs,
        psizes=psizes,
        order=MPI.ORDER_C,
    )
    darray_type.Commit()

    # Allocate memory for the local portion of the distributed array
    local_array = np.zeros(local_array_size, dtype="i")

    # Scatter the global array to the local arrays
    comm.Scatterv([global_array, MPI.FLOAT], local_array, root=0)

    # Each process prints its local array
    print(f"Rank {rank}: Local array {local_array}")

    # Clean up the datatype
    darray_type.Free()


if __name__ == "__main__":
    main()
