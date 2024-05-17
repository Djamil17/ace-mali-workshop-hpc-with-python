import numpy as np
from mpi4py import MPI
from mpi4py.util import dtlib


def block_partition(matrix: np.array, n: int) -> np.array:
    """
    Takes a square matrix of dimensions mxm and breaks into
    square submatrices of dimensions nxn.

    Parameters:
    - matrix: 2d np.array matrix to be divided
    - n: dimension of the matrix

    Returns:
    np.array with dimensions (mxm/(nxn),n,n)

    Example:

    """
    m = matrix.shape[0]
    blocks = []
    for i in range(0, m, n):
        for j in range(0, m, n):
            block = matrix[i : i + n, j : j + n]
            blocks.append(block)
    return np.array(blocks)


def unblock(block_matrix: np.array, n: int) -> np.array:
    """
    Takes a square matrix of dimensions mxm and breaks into
    square submatrices of dimensions nxn.

    Parameters:
    - matrix: 3d np.array matrix to be collapsed
    - n: dimension of the matrix

    Returns:
    np.array with dimensions (nxn)

    Example:

    """
    b, m, _ = block_matrix.shape
    matrix = np.empty((n, n))
    for i in range(0, b):
        for j in range(0, m):
            for k in range(0, m):
                matrix[j, k] = block_matrix[i, j, k]
    return matrix


def transpose(matrix: np.array) -> np.array:
    """
    Takes a matrix of dimensions mxn and gives the
    tranpose nxm.

    Parameters:
    - matrix: 2d np.array matrix to be tranposed

    Returns:
    np.array with dimensions (nxm)

    Example:

    """
    m, n = matrix.shape
    new_matrix = np.empty((n, m))
    for row in range(m):
        new_matrix[:, row] = matrix[row, :]
    return new_matrix


def main():
    ROOT = 0
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    dtype = MPI.FLOAT
    np_dtype = dtlib.to_numpy_dtype(dtype)
    itemsize = dtype.Get_size()

    dim = None
    blocks = None
    local_block = None
    if rank == 0:
        print("dimensions of the square matrix:")
        dim = int(input())
        print("block dimension:")
        block_dimension = int(input())
        A = np.array([np.arange(dim) for _ in range(dim)])
        blocks = block_partition(A, block_dimension)
        final_buf = np.empty((blocks.shape), dtype=np_dtype)
        try:
            if dim % size != 0 and dim % 2 != 0 and size % 2 != 0:
                raise ValueError(
                    f"Dimensions {dim} cannot be block partitioned across {size}\
processes and dimensions and size are not even numbers"
                )
        except ValueError as e:
            print(e)
            exit(1)

    local_block = comm.scatter(blocks, root=ROOT)
    dim = comm.bcast(dim, root=ROOT)
    block_dimension = local_block.shape[1]
    win_size = dim**2 * itemsize
    win = MPI.Win.Allocate(win_size, comm=comm)
    local_buf = np.empty((block_dimension, block_dimension), dtype=np_dtype)

    win.Lock(rank=rank)
    local_buf = transpose(local_block)
    win.Put(local_buf, target_rank=rank)
    win.Unlock(rank=rank)
    comm.Barrier()

    if rank == 0:
        win.Lock(rank=ROOT)
        for i in range(0, size):
            win.Get(local_buf, target_rank=i)
            final_buf[i] = local_buf
        win.Unlock(rank=ROOT)

    if rank == 0:
        print(final_buf)

    win.Free()


if __name__ == "__main__":
    main()