"""
This program demonstrates builds an asynchronous bcast, scatter, and gather
and demonstrates their use via vector-vector addition.

Usage:
    Run the script with an MPI execution command such as
   `mpiexec -n 2 python script_name.py`.

Date: 05/23/2024
Author: Djamil Lakhdar-Hamina

"""

from itertools import islice
from typing import Generator, Iterable, List, Sequence, Type

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def part_gen(
    iterable: Iterable, iterable_type: Type, n: int or float
) -> "Generator[Iterable]":
    """
    Takes an iterable and breaks it up into iterable_type iterators of n size .

    Parameters:
    - iterable: the iterator to break up
    - iterable_type : the type of iterator to be broken up into
    - n : the number of elements in the iterator

    Returns:
    A list of iterators with number of elements n

    """
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := iterable_type(islice(it, n)):
        yield batch


def partition(
    n: int,
    inner_iterable: Iterable,
    outer_iterable_type: Iterable = List,
    inner_iterable_type: Iterable = List,
) -> Iterable[List]:
    """
    Partition an iterable into chunks of a specified size.

    This function divides an input iterable into smaller chunks, each of size `n`.
    The resulting chunks are stored in an iterable specified by `outer_iterable`,
    and each chunk itself is of the type specified by `inner_iterable_type`.

    Parameters:
    n (int): The size of each chunk.
    inner_iterable (iter): The iterable to be partitioned.
    outer_iterable (iter): The type of the outer iterable that will hold the chunks.
    Default is `list`. inner_iterable_type (iter): The type of the inner iterables that
    represent the chunks. Default is `list`.

    Returns:
    iter[list]: An iterable containing the partitioned chunks.

    Example:
    >>> list(partition(3, range(10)))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    >>> tuple(partition(2, 'abcdef', outer_iterable=tuple, inner_iterable_type=str))
    (('a', 'b'), ('c', 'd'), ('e', 'f'))
    """
    return outer_iterable_type(part_gen(inner_iterable, inner_iterable_type, n))


def async_bcast(
    buf: object, buf_size: int, dtype: Type, comm: "MPI.Comm", size: int, root: int = 0
) -> object:
    """
    Asynchronously broadcasts data from the root process to all other processes in the
    given MPI communicator.

    Parameters:
    buf (object): The buffer containing the data to be broadcast. This is only used by
    the root process.
    buf_size (int): The size of the buffer (number of elements) to be broadcast.
    dtype (Type): The data type of the elements in the buffer.
    comm (MPI.Comm): The MPI communicator over which the data is to be broadcast.
    size (int): The total number of processes in the communicator.
    root (int, optional): The rank of the root process from which data will be
    broadcast. Default is 0.

    Returns:
    object: The buffer containing the received data on non-root processes. On the root
    process, this is the original buffer.

    Example:
    >>> from mpi4py import MPI
    >>> import numpy as np
    >>> comm = MPI.COMM_WORLD
    >>> rank = comm.Get_rank()
    >>> size = comm.Get_size()
    >>> if rank == 0:
    >>>     buf = np.array([1, 2, 3, 4], dtype=np.int)
    >>> else:
    >>>     buf = None
    >>> recvbuf = async_bcast(buf, 4, np.int, comm, size, root=0)
    >>> print(f"Process {rank} received: {recvbuf}")

    Notes:
    - This function uses non-blocking send (Isend) and receive (Irecv) operations to
    achieve asynchronous communication.
    - The root process sends the buffer to all other processes.
    - Non-root processes allocate an empty buffer of the specified size and type to
      receive the data.
    - The function returns the received buffer on non-root processes and the original
    buffer on the root process.
    """

    if rank == root:
        recvbuf = buf
        requests = []
        for p in range(0, size):
            if p != root:
                request = comm.Isend(buf=recvbuf, dest=p, tag=0)
                requests.append(request)
        MPI.Request.Waitall(requests)
    else:
        recvbuf = np.empty(buf_size, dtype=dtype)
        request = comm.Irecv(buf=recvbuf, source=0, tag=0)
        request.Wait()
    return recvbuf


def async_scatter(
    sendbuf: Sequence[object],
    recv_size: int,
    dtype: Type,
    comm: "MPI.Comm" = comm,
    size: int = size,
    root: int = 0,
) -> object:
    """
    Asynchronously scatters data from the root process to all other processes in the
    given MPI communicator.

    Parameters:
    sendbuf (Sequence[object]): A sequence of buffers, where each buffer contains the
    data to be sent to a specific process. This is only used by the root process.
    recv_size (int): The size of the buffer (number of elements) to be received by each
    process.
    dtype (Type): The data type of the elements in the buffer.
    comm (MPI.Comm): The MPI communicator over which the data is to be scattered.
    size (int): The total number of processes in the communicator.
    root (int, optional): The rank of the root process from which data will be
    scattered. Default is 0.

    Returns:
    object: The buffer containing the received data on each process.

    Example:
    >>> from mpi4py import MPI
    >>> import numpy as np
    >>> comm = MPI.COMM_WORLD
    >>> rank = comm.Get_rank()
    >>> size = comm.Get_size()
    >>> if rank == 0:
    >>>     sendbuf = [np.array([i, i+1, i+2], dtype=np.int) for i in range(size)]
    >>> else:
    >>>     sendbuf = None
    >>> recvbuf = async_scatter(sendbuf, 3, np.int, comm, size, root=0)
    >>> print(f"Process {rank} received: {recvbuf}")

    Notes:
    - This function uses non-blocking send (Isend) and receive (Irecv) operations to
    achieve asynchronous communication.
    - The root process sends a specific portion of the send buffer to each process.
    - Non-root processes allocate an empty buffer of the specified size and type to
    receive their portion of the data.
    - The function returns the received buffer on all processes.
    """
    if rank == root:
        recvbuf = sendbuf[0]
        requests = []
        for p in range(0, size):
            if p != root:
                request = comm.Isend(sendbuf[p], p, p)
                requests.append(request)
        MPI.Request.Waitall(requests)
    else:
        recvbuf = np.empty(recv_size, dtype=dtype)
        request = comm.Irecv(recvbuf, source=root, tag=rank)
        request.Wait()
    return recvbuf


def async_gather(
    sendbuf: Sequence[object],
    recv_size: int,
    dtype: Type,
    comm: "MPI.Comm" = comm,
    size: int = size,
    root: int = 0,
) -> List[object]:
    """
    Asynchronously gathers data from all processes to the root process in the given MPI
    communicator.

    Parameters:
    sendbuf (Sequence[object]): The buffer containing the data to be sent from each
    process.
    recv_size (int): The size of the buffer (number of elements) to be received
    from each process.
    dtype (Type): The data type of the elements in the buffer.
    comm (MPI.Comm): The MPI communicator over which the data is to be gathered.
    size (int): The total number of processes in the communicator.
    root (int, optional): The rank of the root process to which data will be gathered.
    Default is 0.

    Returns:
    List[object]: A list of buffers containing the received data on the root process.
    On non-root processes, returns None.

    Example:
    >>> from mpi4py import MPI
    >>> import numpy as np
    >>> comm = MPI.COMM_WORLD
    >>> rank = comm.Get_rank()
    >>> size = comm.Get_size()
    >>> sendbuf = np.array([rank, rank + 1, rank + 2], dtype=np.int)
    >>> gathered_data = async_gather(sendbuf, 3, np.int, comm, size, root=0)
    >>> if rank == 0:
    >>>     for i, data in enumerate(gathered_data):
    >>>         print(f"Process {i} sent: {data}")

    Notes:
    - This function uses non-blocking send (Isend) and receive (Irecv) operations to
    achieve asynchronous communication.
    - The root process receives data from all processes and gathers it into a list.
    - Non-root processes send their data to the root process.
    - The function returns a list of received buffers on the root process, and returns
      None on non-root processes.
    """

    if rank == root:
        requests = []
        gather_arr = []
        gather_arr.append(sendbuf)
        for p in range(0, size):
            if p != root:
                recv_buf = np.empty(recv_size, dtype=dtype)
                request = comm.Irecv(recv_buf, p, p)
                requests.append(request)
                gather_arr.append(recv_buf)
        MPI.Request.Waitall(requests)
        return gather_arr
    else:
        request = comm.Isend(sendbuf, dest=root, tag=rank)
        request.Wait()


def main():
    assert size > 2

    if rank == 0:
        print("input vector length:")
        n = int(input())
        chunk = n // (size)
        chunk_buffer = np.array([chunk], dtype="i")
        # print(f"number elements per process: {chunk}")
        try:
            if n % size != 0:
                raise ValueError(
                    "The values are not evenly partitioned between n workers\
processors"
                )
        except ValueError as e:
            print(e)
            comm.Abort()
        # define 2 random vectors
        x, y = np.random.rand(n), np.random.rand(n)
        partitions = partition(chunk, x, list, list), partition(chunk, y, list, list)
        # split each vector into p chunks of n size
        sendbuf = np.array(
            [np.concatenate((part_x, part_y)) for (part_x, part_y) in zip(*partitions)]
        )
        # print(f"split it up and concatenated, sendbuf is : \n {sendbuf}\n")

    else:
        n = chunk_buffer = sendbuf = None

    # broadcast chunk length
    buf_size = 1  # Size of the buffer
    chunk = async_bcast(
        buf=chunk_buffer, buf_size=buf_size, dtype="i", comm=comm, size=size
    )[0]
    recvbuf = async_scatter(
        sendbuf=sendbuf, recv_size=chunk * 2, dtype=float, comm=comm, size=size
    )
    local_sum = recvbuf[:chunk] + recvbuf[chunk:]
    finalrecvbuf = np.array(
        async_gather(sendbuf=local_sum, recv_size=chunk, dtype=float, comm=comm)
    )
    if rank == 0:
        assert np.all(x + y == finalrecvbuf.flatten())


if __name__ == "__main__":
    main()
