"""
Write data (rank number) 10 times from a numpy buffer to file in parallel.

Each MPI process writes its rank to a file in a non-contiguous pattern.
The file view is set such that each process writes to a distinct section of the file.
The program demonstrates parallel I/O using mpi4py with custom data types.

Date: 05/19/2024
Author: Djamil Lakhdar-Hamina
"""

import numpy as np
from mpi4py import MPI


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    status = MPI.Status()

    # set file mode then open the file handler
    filename = "./.noncontig_read_write.txt"
    amode = MPI.MODE_CREATE | MPI.MODE_RDWR | MPI.MODE_APPEND
    fh = MPI.File.Open(comm, filename, amode)

    item_count = 10

    buffer = np.empty(item_count, dtype="i")
    buffer[:] = rank

    # create and commit custom data structure
    filetype = MPI.INT.Create_vector(item_count, 1, size)
    filetype.Commit()

    displacement = MPI.INT.Get_size() * item_count * rank
    offset = MPI.INT.Get_size() * item_count
    fh.Set_view(displacement, filetype=filetype)
    fh.Write_at_all(offset, buffer, status)

    new_buffer = np.empty(item_count, dtype="i")
    info = fh.Get_view()
    print(info)
    fh.Read_at_all(offset, new_buffer)
    print(new_buffer)

    if rank == 0:
        fh.Delete(filename)

    filetype.Free()
    fh.Close()


if __name__ == "__main__":
    main()
