from mpi4py import MPI


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        vector = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    else:
        vector = None
        total_sum = None

    local_vector = comm.bcast(vector, root=0)
    local_sum = 0
    for i in local_vector:
        local_sum += i
    total_sum = comm.reduce(local_sum, op=MPI.SUM)
    if rank == 0:
        print(total_sum)


main()
