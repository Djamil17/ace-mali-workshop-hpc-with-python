"""
This program demonstrates basic MPI (Message Passing Interface) asynchronous
communication usingthe mpi4py library. The example involves two processes
(ranks 0 and 1) exchanging messages.

Usage:
    Run the script with an MPI execution command such as
   `mpiexec -n 2 python script_name.py`.

Date: 05/23/2024
Author: Djamil Lakhdar-Hamina

"""

from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    statuses = [MPI.Status() for _ in range(size // 2 + 1)]
    assert size == 2
    # non numpy-example
    if rank == 0:
        data = bytearray("Hello, world!".encode())
        request = comm.Isend(data, dest=1)
        print(f"rank {rank}: line 28 {request.Test(statuses[0])}")
        print(f"rank {rank} :line 29 {2 + 3} ")
    elif rank == 1:
        buf = bytearray(20)
        request = comm.Irecv(buf, source=0)
        print(f"rank {rank}: line 33 {request.Test(statuses[0])}")
        print(f"rank {rank} : line 34 {6 + 3} ")
        request.Wait(status=statuses[0])
        print(f"rank {rank}: line 36 {request.Test(statuses[0])}")
        print(f"{buf.decode()} From process {rank}")


if __name__ == "__main__":
    main()
