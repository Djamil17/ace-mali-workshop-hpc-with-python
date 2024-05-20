import subprocess

from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    status = MPI.Status()

    amode = MPI.MODE_CREATE | MPI.MODE_RDWR | MPI.MODE_APPEND
    fh = MPI.File.Open(comm=comm, filename="./.read_write.txt", amode=amode)
    message = f"Hello world, from process {rank}!\n".encode()
    chunk_size = f"Hello world, from process {rank}!\n".__sizeof__()
    offset = chunk_size * rank
    fh.Write_at_all(offset, message, status)

    if rank == 0:
        # check if file exists by printing to stdout
        subprocess.run(["cat", "test.txt"])

    buffer = bytearray(chunk_size)
    fh.Read_at_all(offset, buffer)

    # Gather all buffers
    all_buffer = comm.gather(buffer, root=0)
    # Print from root in order
    if rank == 0:
        print("\n")
        for buffer in reversed(all_buffer):
            print(buffer.decode(), end="")

    fh.Close()


main()
