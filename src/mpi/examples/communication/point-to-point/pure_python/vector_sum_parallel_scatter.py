"""
Shards two vectors into two disjoint set of pieces, scatters the shards, adds the shards
then reassembles the shards into an overall vector.

Date: 05/16/2024
Author: Djamil Lakhdar-Hamina

"""

from itertools import chain, islice
from random import randint
from typing import Generator, Iterable, List, Type

from mpi4py import MPI


def part_gen(iterable: Iterable, iterator_type: Type, n: int) -> Generator:
    """
    Takes an iterable and breaks it up into iterable_type iterators of n size .

    Parameters:
    - iterable: the iterator to break up
    - iterable_type : the type of iterator to be broken up into
    - n : the number of elements in the iterator

    Returns:
    A generator of iterators with number of elements n

    """
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := iterator_type(islice(it, n)):
        yield batch


def partition(
    iterable: Iterable, outer_iterator_type: Type, inner_iterator_type: Type, n: int
) -> Iterable[Iterable]:
    """
    Takes an iterable and breaks it up into iterable_type iterators of n size .

    Parameters:
    - iterable: the iterator to break up
    - outer_iterator_type: the iterator to wrap inner pieces in
    - iterable_type : the type of iterator to be broken up into
    - n : the number of elements in the iterator

    Returns:
    A actual iterator of iterators with number of elements n

    Example:

    >>> partition([1,2,3,4,5,6],tuple, list,2)
        ([1,2],[3,4],[5,6])

    """
    return outer_iterator_type(part_gen(iterable, inner_iterator_type, n))


def flatten(outer_iterable: Iterable, iterable: Iterable[Iterable]) -> Iterable:
    """
    Flatten an iterable into a one-dimensional outer_itreable.

    Parameters:
    - outer_iterable: the flattened iterator type
    - iterable: the iterator of iterators to be flattened

    Returns:
    An actual one dimensioal iterator.

    Example:

    >>> flatten(list, [[1,2,3],[1,2]])
    [1,2,3,1,2]

    """
    return outer_iterable(chain.from_iterable(iterable))


def rand_vector(lower_limit: int, upper_limit: int, n: int) -> List[int]:
    """ "
    Produces a random vector whose elements are between
    lower limit and upper limit of length n.

    Parameters:
    - lower_limit: lower limit of element
    - upper_limit: upper limit of element

    Returns:
    list of integers
    """
    return [randint(lower_limit, upper_limit) for _ in range(0, n)]


def produce_zero_vector(n: int) -> List[int]:
    """ "
    Produces a vector of zeros of length n

    Parameters:
    - n: length of vector

    Returns:
    list of zeros
    """
    return [0 for _ in range(0, n)]


def vector_sum(x: List, y: List) -> List:
    """
    Add two lists (vectors) element wise.

    Parameters:
    - x : first vector of size n
    - y : second vector of size (n) or else program fails

    Returns:
    a vector or list representing x+y element-wise
    """

    try:
        if len(x) != len(y):
            raise ValueError("the length of x and y do not match")
        if type(x[0]) is not type(y[0]):
            raise ValueError("the type of x and y do not match")
        z = [0 for _ in range(0, len(x))]
    except ValueError as e:
        print(e)

    for i in range(0, len(x)):
        z[i] = x[i] + y[i]

    return z


def main():
    # define some constants
    ROOT = 0
    LOWER_LIMIT = 1
    UPPER_LIMIT = 10
    VECTOR_LENGTH = 10

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    assert size > 1

    try:
        if rank == 0:
            if VECTOR_LENGTH % size != 0:
                raise ValueError(
                    "elements are not evenly partitioned between\
    processors"
                )
    except ValueError as e:
        print(e)

    chunk = VECTOR_LENGTH // size

    if rank == 0:
        # produce two random vectors
        x = rand_vector(LOWER_LIMIT, UPPER_LIMIT, VECTOR_LENGTH)
        y = rand_vector(LOWER_LIMIT, UPPER_LIMIT, VECTOR_LENGTH)

        # partition them or shard them, one shard for each process including root!
        partitioned_x = partition(x, list, list, chunk)
        partitioned_y = partition(y, list, list, chunk)

        # build a list of tuples, the first element is x chunk, second y chunk
        x_y = [
            (x_chunk, y_chunk)
            for (x_chunk, y_chunk) in zip(partitioned_x, partitioned_y)
        ]
    else:
        x_y = None

    x_y = comm.scatter(x_y, root=ROOT)
    local_x, local_y = x_y[0], x_y[1]
    sum = vector_sum(local_x, local_y)
    result = comm.gather(sum, root=0)
    if rank == 0:
        assert flatten(list, result) == vector_sum(x, y)
        print(f"{x} + {y} =", flatten(list, result))

    return result


if __name__ == "__main__":
    main()
