import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Define the global array dimensions
global_rows = 8
global_cols = 8

# Create a cartesian communicator
dims = MPI.Compute_dims(size, 2)  # Assuming 2D grid of processes
periods = [False, False]  # Non-periodic grid
reorder = True
cart_comm = comm.Create_cart(dims, periods=periods, reorder=reorder)

# Define the distribution
dist = [MPI.DISTRIBUTE_BLOCK, MPI.DISTRIBUTE_BLOCK]
dargs = [MPI.DISTRIBUTE_DFLT_DARG, MPI.DISTRIBUTE_DFLT_DARG]
psizes = dims

# Create the distributed array
array = MPI.Create_dist_blocked((global_rows, global_cols), dist, dargs, psizes)

# Determine local array size and allocate local array
local_rows = array.shape[0]
local_cols = array.shape[1]
local_array = np.zeros((local_rows, local_cols))

print(f"Process {rank} has array segment:\n{local_array}")

# Finalize MPI
MPI.Finalize()
