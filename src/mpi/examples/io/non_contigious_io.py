"""
Write data (rank number multiplied size times) from buffer to
file in parallel.

Date: 05/19/2024
Author: Djamil Lakhdar-Hamina
"""

import numpy as np
from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    status = MPI.Status()

    # set file mode then open the file handler
    amode = MPI.MODE_CREATE | MPI.MODE_RDWR | MPI.MODE_APPEND
    fh = MPI.File.Open(comm, "./.datafile.noncontig", amode)

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
    fh.Get_view()
    fh.Read_at_all(offset, new_buffer)
    print(new_buffer)

    filetype.Free()
    fh.Close()


if __name__ == "__main__":
    main()
