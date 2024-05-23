"""
Use numpy array and mpi4py to add two vectors together.

Date: 05/22/2024
Author: Djamil Lakhdar-Hamin
"""

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

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("input vector length:")
        n = int(input())
        chunk = n // (size)
        info = np.array([n, chunk], dtype=int)
        print(f"number elements per process: {chunk}")
        try:
            if n % size != 0:
                raise ValueError(
                    "The values are not evenly partitioned between n workers\
processors"
                )
        except ValueError as e:
            print(e)
            comm.Abort()
        # define 2 random vectors
        x, y = np.random.rand(n), np.random.rand(n)
        partitions = np.array(partition(x, chunk), dtype=float), np.array(
            partition(y, chunk), dtype=float
        )
        # split each vector into p chunks of n size
        sendbuf = np.array(
            [np.concatenate((part_x, part_y)) for (part_x, part_y) in zip(*partitions)]
        )
        print(f"split it up and concatenated, sendbuf is : \n {sendbuf}\n")
    else:
        info = np.zeros(2, dtype=int)
        sendbuf = None

    comm.Bcast([info, MPI.INT], root=0)
    n, chunk = info
    recvbuf = np.zeros(chunk * 2, dtype=float)

    comm.Scatter([sendbuf, MPI.FLOAT], [recvbuf, MPI.FLOAT], root=ROOT)
    midpoint = len(recvbuf) // 2
    local_sum = recvbuf[:midpoint] + recvbuf[midpoint:]

    finalrecvbuf = np.zeros((size, chunk))
    comm.Gather([local_sum, MPI.FLOAT], [finalrecvbuf, MPI.FLOAT], root=ROOT)
    if rank == 0:
        print(f" {x} + {y} = {finalrecvbuf.flatten()}")
        assert np.all(x + y == finalrecvbuf.flatten())


if __name__ == "__main__":
    main()
