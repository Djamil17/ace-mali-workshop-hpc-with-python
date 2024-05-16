import numpy as np
from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Define the data to be scattered from the root process
    if rank == 0:
        send_data = np.array(
            [[i + j for j in range(size)] for i in range(size)], dtype=int
        )
        print(send_data.shape)
    else:
        send_data = None

    # Allocate memory for receive buffer in each process
    recv_data = np.empty(size, dtype=int)

    # Scatter data from root process to all other processes
    comm.Scatter(send_data, recv_data, root=0)

    print(f"Process {rank} received scattered data: {recv_data}")

    # Perform some computation (e.g., sum the received data)
    local_sum = recv_data * 12

    # Gather results back to the root process
    if rank == 0:
        global_sums = np.empty((3, size), dtype=int)
    else:
        global_sums = None

    comm.Gather(local_sum, global_sums, root=0)

    if rank == 0:
        print("Global sums gathered by root process:")
        print(global_sums)


main()
