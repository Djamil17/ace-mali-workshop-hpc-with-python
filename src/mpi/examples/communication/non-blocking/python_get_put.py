# TODO: need to make this work

from mpi4py import MPI


def empty(n: int) -> list:
    "return empty list , list with 0s, of length n"
    return [0 for _ in range(n)]


def inplace_fill(arr: list, value: int or float or str) -> list:
    "replace elements in list with a value"
    length = len(arr)
    for i in range(length):
        arr[i] = value
    return arr


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    N = 42
    dtype = MPI.FLOAT
    dtype_size = dtype.Get_size()
    byte_size = N * dtype_size if rank == 0 else 0

    win = MPI.Win.Allocate(byte_size, comm=comm)
    buf = empty(N)

    if rank == 0:
        inplace_fill(buf, value=42)
        win.Lock(rank=0)
        byte_buf = bytearray(buf)
        win.Put([byte_buf, MPI.FLOAT], target_rank=0)
        win.Unlock(rank=0)
        comm.Barrier()
    else:
        comm.Barrier()
        win.Lock(rank=0)
        byte_buf = bytearray(N * dtype_size)
        win.Get([buf, MPI.FLOAT], target_rank=0)
        win.Unlock(rank=0)
        assert all(val == 42 for val in buf)

    if rank == 0:
        print("all elements in buffer are 42!")


main()
