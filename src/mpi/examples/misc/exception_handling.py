from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    assert size % 2 == 0
    # try:
    #     size=comm.Get_size()
    #     if size % 2 != 0:
    #         MPI.Finalize()
    #         raise ValueError("The size is not even...")
    # except ValueError as e:
    #     print(e)


main()

# notice
