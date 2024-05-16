import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# set file mode then open the file handler
amode = MPI.MODE_CREATE | MPI.MODE_RDWR | MPI.MODE_APPEND
fh = MPI.File.Open(comm, "./datafile.noncontig", amode)

item_count = 10

buffer = np.empty(item_count, dtype="i")
buffer[:] = rank

# create and commit custom data structure
filetype = MPI.INT.Create_vector(item_count, 1, size)
filetype.Commit()

displacement = MPI.INT.Get_size() * rank
fh.Set_view(displacement, filetype=filetype)
fh.Write_all(buffer)

filetype.Free()
fh.Close()
