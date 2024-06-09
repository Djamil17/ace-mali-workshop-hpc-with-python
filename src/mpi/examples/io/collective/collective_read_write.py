"""
Demonstrates collective read and write operations using mpi4py.

This program performs the following steps:
1. Initializes MPI and obtains the rank of each process.
2. Each process prepares a unique message and calculates its offset.
3. All processes collectively write their messages to distinct offsets in
the same file using `Write_at_all`.
4. Each process reads its own message from the file using `Read_at_all`.
5. The read messages are gathered at the root process and printed in order.
6. The file is deleted after the operations are complete.

The root process coordinates the output messages to indicate the progress
 of write and read operations.

Date: 06/02/2024
Author: Djamil Lakhdar-Hamina
"""

from mpi4py import MPI


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    status = MPI.Status()

    filename = "./.collective_read_write.txt"
    amode = MPI.MODE_CREATE | MPI.MODE_RDWR | MPI.MODE_APPEND
    fh = MPI.File.Open(comm=comm, filename=filename, amode=amode)
    message = f"Hello world, from process {rank}!\n".encode()
    chunk_size = message.__sizeof__()
    offset = chunk_size * rank

    # write section
    if rank == 0:
        print("now writing...")
    fh.Write_at_all(offset, message, status)

    # read section
    buffer = bytearray(chunk_size)
    fh.Read_at_all(offset, buffer)
    # Gather all buffers
    all_buffer = comm.gather(buffer, root=0)
    # Print from root in order
    if rank == 0:
        print("now reading...")
        for buffer in reversed(all_buffer):
            print(buffer.decode(), end="")
        fh.Delete(filename=filename)
    fh.Close()


if __name__ == "__main__":
    main()
