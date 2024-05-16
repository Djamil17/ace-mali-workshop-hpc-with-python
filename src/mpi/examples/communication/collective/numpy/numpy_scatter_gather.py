from itertools import islice

import numpy as np
from mpi4py import MPI


def part_gen(iterable: iter, n: int) -> list[list]:
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


def partition(iterable: iter, n: int) -> list[list]:
    "partition an iterable into chunks of n size"
    return np.array(list(part_gen(iterable, n)))


def main():
    ROOT = 0
    VECTOR_LENGTH = 10

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    chunk = VECTOR_LENGTH // (size)

    np.random.seed(19)

    sendbuf = None
    recvbuf = np.empty(chunk * 2)

    if rank == 0:
        try:
            if VECTOR_LENGTH % size != 0:
                raise ValueError(
                    "The values are not evenly partitioned between n-1 worker\
processors"
                )
        except ValueError as e:
            print(e)
            comm.Abort()
        # define 2 vectors
        x, y = np.random.rand(VECTOR_LENGTH), np.random.rand(VECTOR_LENGTH)
        partitions = np.array(partition(x, chunk)), np.array(partition(y, chunk))
        # split each vector into p chunks of n size
        sendbuf = np.array(
            [np.concatenate((part_x, part_y)) for (part_x, part_y) in zip(*partitions)]
        )
        print(f"sendbuf is : \n {sendbuf}\n")

    comm.Scatter(sendbuf, recvbuf, root=ROOT)
    midpoint = len(recvbuf) // 2
    local_sum = recvbuf[:midpoint] + recvbuf[midpoint:]

    finalrecvbuf = None
    if rank == 0:
        finalrecvbuf = np.empty((size, chunk))

    comm.Gather(local_sum, finalrecvbuf, root=ROOT)

    if rank == 0:
        print(f" {x} + {y} = {finalrecvbuf.flatten()}")
        assert np.all(x + y == finalrecvbuf.flatten())


main()
