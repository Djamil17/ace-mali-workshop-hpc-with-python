import numpy as np
from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    status = MPI.Status()

    N = 10
    # non numpy-example
    if rank == 0:
        data = bytearray("Hello, world!".encode())
        request = comm.Isend(data, dest=1)
        print(2 + 3)
    else:
        buf = bytearray(54)
        request = comm.Irecv(buf, source=0)
        print(2 + 3)
        request.wait(status=status)
        print(f"{buf.decode()} From process {rank}")

    # numpy-example
    if rank == 0:
        data = np.random.rand(N)
        request = comm.Isend(data, dest=1)
        print(2 + 3)
    else:
        buf = np.empty(N)
        request = comm.Irecv(buf, source=0)
        print(2 + 3)
        request.wait(status=status)
        print(f"from process {rank}: {buf}")


if __name__ == "__main__":
    main()
