"""
Shows the use of Op class to define user reduction operation
A single random integer is produced across processes, the same one,
then the magnitude is found.

Date: 05/16/2024
Author: Djamil Lakhdar-Hamina

"""

from random import randint, seed

from mpi4py import MPI


def magnitude(x, y, float) -> float:
    return (x**2 + y**2) ** (1 / 2)


Magnitude = MPI.Op.Create(magnitude, True)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    LOWER_LIMIT = 1
    UPPER_LIMIT = 10

    seed(42)

    random_int = randint(LOWER_LIMIT, UPPER_LIMIT)
    print(f"rank {rank}: {random_int}")
    global_magnitude = comm.reduce(random_int, op=Magnitude, root=0)

    if rank == 0:
        print("the global magnitude is:", global_magnitude)
        assert global_magnitude == 2 * random_int

    Magnitude.Free()


if __name__ == "__main__":
    main()
