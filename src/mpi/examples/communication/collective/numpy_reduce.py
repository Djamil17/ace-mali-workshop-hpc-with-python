from random import randint

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

LOWER_LIMIT = 1
UPPER_LIMIT = 10

randomn_int = randint(LOWER_LIMIT, UPPER_LIMIT)
global_sum = comm.reduce(randomn_int, op=MPI.SUM, root=0)

if rank == 0:
    print("the global sum is:", global_sum)
