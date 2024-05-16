import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

assert size > 1

nbytes = MPI.INT.Get_size() // size
custom_dtype = MPI.INT.Create_vector(size, nbytes, 0)
custom_dtype.Commit()
buf = np.empty(size, dtype="i")

if rank == 0:
    buf.fill(1)
    comm.Send([buf, custom_dtype], dest=1)
else:
    comm.Recv([buf, custom_dtype], source=0)
    print(buf)
