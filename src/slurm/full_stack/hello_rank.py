from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if rank == 0:
    for p in range(1, size):
        comm.send(f"Hello {p}!", dest=p)
else:
    message = comm.recv()
    print(message)
