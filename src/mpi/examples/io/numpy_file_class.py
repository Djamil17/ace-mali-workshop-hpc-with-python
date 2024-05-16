from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    status = MPI.Status()

    amode = MPI.MODE_CREATE | MPI.MODE_RDWR | MPI.MODE_APPEND
    fh = MPI.File.Open(comm=comm, filename="./test.txt", amode=amode)
    message = f"Hello world, from process {rank}!\n".encode()
    chunk_size = f"Hello world, from process {rank}!\n".__sizeof__()
    offset = chunk_size * rank
    # fh.Write_ordered(message, status)
    fh.Write_at_all(offset, message, status)

    buffer = bytearray(chunk_size)
    fh.Read_at_all(offset, buffer)

    # Synchronize all processes to ensure ordered printing

    # Print from each process in order
    for r in range(size):
        if rank == r:
            print(buffer.decode())

    fh.Close()


main()
