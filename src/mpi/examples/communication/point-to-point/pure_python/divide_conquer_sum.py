from mpi4py import MPI


def divide_conquer_sum(begin: int = 2, end: int = 10):
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size() - 1

    global_sum = 0
    length = end - begin
    portions = length // size

    try:
        if length % 2 != 0:
            raise ValueError("The range is not an even number...")
        if portions % 2 != 0:
            raise ValueError(
                "The ranges are not evenly allocated between processors..."
            )
    except ValueError as e:
        print(e)

    ranges = [
        range(begin + step * portions, begin + (step + 1) * portions)
        for step in range(0, portions + 2)
    ]

    if rank == 0:
        for i in range(1, size + 1):
            comm.send(obj=ranges[i - 1], dest=i)
    else:
        data = sum(comm.recv(source=0))
        comm.send(data, dest=0)

    if rank == 0:
        for i in range(1, size + 1):
            data = comm.recv(source=i)
            global_sum += data
        print(global_sum)

    return global_sum


divide_conquer_sum(2, 10)
