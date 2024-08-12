""""
Scatters and gathers a list

Date: 05/16/2024
Author: Djamil Lakhdar-Hamina

"""

from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    foo = [1, 2, 3, 4, 5]
    if rank == 0:
        print(f"Sending chunks of {foo} from root...")
    local_foo = comm.scatter(foo, root=0)
    print(f"Sent {local_foo} to {rank}")
    comm.Barrier()

    new_foo = comm.gather(local_foo, root=0)
    if rank == 0:
        print(f"Received and reassembled {new_foo}")


if __name__ == "__main__":
    main()
