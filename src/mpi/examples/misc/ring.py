from mpi4py import MPI

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

assert size > 1

# send from 0 -> receive from 1 -> send from 1 -> receive from 2
token = 0
if rank == 0 and token == 0:
    print(f"token is at {rank} and is {token}")
    token = -1
    comm.send(token, dest=rank + 1)

elif rank > 0 and rank < size - 1:
    token = comm.recv(source=rank - 1)
    comm.send(token, dest=rank + 1)
    print(f"token is at {rank} and is {token}")
elif rank == size - 1:
    token = comm.recv(source=rank - 1)
    comm.send(token, dest=0)
    print(f"token is at {rank} and is {token}")

comm.Barrier()

if rank == 0 and token == -1:
    print("token is back at 0 and is -1!")
    comm.recv(source=rank - 1)
