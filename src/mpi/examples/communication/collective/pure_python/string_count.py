from random import sample

from mpi4py import MPI


def main() -> None:
    NUCLEOTIDES = "agtc"
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    length = 1000
    long_string = [sample(NUCLEOTIDES, k=1)[0] for _ in range(length)]
    chunk_size = length // size
    local_string = long_string[rank * chunk_size : (rank + 1) * chunk_size]
    length_local_string = len(local_string)
    lengths = comm.gather(length_local_string)

    if rank == 0:
        sum = 0
        for i in lengths:
            sum += i
            print(sum)


if __name__ == "__main__":
    main()
