import numpy as np
from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    status = MPI.Status()

    np.random.seed(1917)

    amode = MPI.MODE_CREATE | MPI.MODE_RDWR | MPI.MODE_APPEND
    fh = MPI.File.Open(comm=comm, filename="./numpy_test.txt", amode=amode)
    buffer = np.random.rand(size) * rank
    chunk_size = buffer.nbytes
    offset = chunk_size * rank
    fh.Write_at_all(offset, buffer, status)

    buffer = np.empty(size, dtype=float)
    fh.Read_at_all(offset, buffer)

    # Gather all buffers
    all_buffer = comm.gather(buffer, root=0)
    # Print from root in order
    if rank == 0:
        print("\n")
        for buffer in reversed(all_buffer):
            print(buffer, end="\n")

    fh.Close()


main()
