"""
Program function to demonstrate point-to-point file I/O operations using mpi4py.

Each MPI process writes its own message to a unique offset in a shared file.
The processes then read their respective messages from the file and print them.

Date: 06/02/2024
Author: Djamil Lakhdar-Hamina

"""
from mpi4py import MPI


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    filename = "./.point_read_write.txt"
    amode = MPI.MODE_CREATE | MPI.MODE_WRONLY
    fh = MPI.File.Open(comm=comm, filename=filename, amode=amode)
    write_buffer = f"Hello world, from process {rank}!".encode("utf-8")
    buffer_size = write_buffer.__sizeof__()
    offset = buffer_size * rank
    fh.Seek(offset)
    fh.Write(write_buffer)
    fh.Sync()
    fh.Close()

    amode = MPI.MODE_RDONLY
    read_buffer = bytearray(buffer_size)
    fh = MPI.File.Open(comm, filename, amode)
    fh.Seek(offset)
    fh.Read(read_buffer)
    comm.Barrier()
    fh.Close()
    print(read_buffer.decode("utf-8"))


if __name__ == "__main__":
    main()
