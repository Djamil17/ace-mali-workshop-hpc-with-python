from random import randint

from mpi4py import MPI


def vector_sum(x: list, y: list) -> list:
    try:
        if len(x) != len(y):
            raise ValueError("the length of x and y do not match")
        if type(x[0]) is not type(y[0]):
            raise ValueError("the type of x and y do not match")
        z = [0 for _ in range(0, len(x))]
    except ValueError as e:
        print(e)

    for i in range(0, len(x)):
        z[i] = x[i] + y[i]

    return z


def main():
    ROOT = 0
    LOWER_LIMIT = 1
    UPPER_LIMIT = 1000
    VECTOR_LENGTH = 5

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        x_y = [randint(LOWER_LIMIT, UPPER_LIMIT) for _ in range(0, VECTOR_LENGTH)], [
            randint(LOWER_LIMIT, UPPER_LIMIT) for _ in range(0, VECTOR_LENGTH)
        ]
    else:
        x_y = None

    x_y = comm.bcast(x_y, root=ROOT)

    if rank != 0:
        x, y = x_y[0], x_y[1]
        sum = vector_sum(x, y)
        print(f"{sum}")


main()
