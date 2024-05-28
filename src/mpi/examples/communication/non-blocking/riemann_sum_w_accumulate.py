"""
This program computes the Riemann sum of a user-defined function over a specified
interval using MPI for parallel processing. The main components of the program include
functions for partitioning iterables, generating parts of iterables,
validating single-line expressions,creating functions dynamically from user input, and
performing the Riemann sum calculation. The MPI framework is utilized to distribute the
computation across multiple processes.

Functions:
----------
- part_gen(iterable: Iterable,
           iterable_type: Type,
           n: int or float) -> "Generator[Iterable]":
    Takes an iterable and breaks it up into `iterable_type` iterators of size `n`.

- partition(n: int,
 inner_iterable: Iterable, outer_iterable_type: Iterable = List,
 inner_iterable_type: Iterable = List) -> Iterable[List]:
    Partitions an iterable into chunks of a specified size.

- is_single_line(expression: str) -> bool:
    Checks if a string expression is a single line.

- create_function_string_from_user_input() -> str:
    Dynamically creates a single-lined function string from user input.

- create_function_on_process(function_string: str) -> Callable:
    Compiles a function from a function string on a process.

- riemann_sum_left(f: Callable[[(float | int)], float], interval: Tuple[(float | int)],
delta_x: (float | int)) -> float | int:
    Approximates the integral of `f` over an interval using the left Riemann sum method.

MPI Process:
-------------
- main():
    Manages the MPI processes, including collecting user input for the function and
    interval, distributing the computation, and aggregating the results.

Example Usage:
--------------
1. The program prompts the user to enter a single-line function body and the interval
(start, end) and number of partitions.
2. The main function handles the MPI initialization, input distribution,
and the parallel computation of the Riemann sum.
3. The computed integral is printed by the root process.

Note:
-----
- Ensure MPI is set up in your environment to run this program.
- The user input for the function should be in the form of a valid single-line Python
expression.

Date: 05/28/2024
Author : Djamil Lakhdar-Hamina

"""

from itertools import islice
from typing import Callable, Generator, Iterable, List, Tuple, Type

import numpy as np
from mpi4py import MPI
from mpi4py.util import dtlib


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


def is_single_line(expression: str) -> bool:
    """
    Takes a string expression and checks if the string is one-line by counting
    /n (newline) characters.

    Parameters:
    - expression: the function body in string form

    Returns:
    A boolean indicating if function body is single-lined (true)

    """
    # Count the number of newline characters
    newline_count = expression.count("\n")
    # Return True if there is only one newline character, False otherwise
    return newline_count == 0


def create_function_string_from_user_input() -> str:
    """
    Dynamically create a single-lined function string from user input.

    Parameters:

    Returns:
    A single-lined function of form f(x): return expression
    """
    # Get a string from the user
    print("Enter a single-lined function body: ")
    function_string = input()
    # Check that string is well-formed
    assert is_single_line(function_string)
    # Execute code while making sure that scope of f is not just in exec bloc
    return function_string


def create_function_on_process(function_string: str) -> Callable:
    """
    Compiles a function from a function string on a process.

    Parameters:
    - function_string: string or body of function

    Returns:
    The function with body given

    """
    locals_dict = {}
    exec(f"def f(x): return {function_string}", {}, locals_dict)
    return locals_dict["f"]


def riemann_sum_left(
    f: Callable[[(float | int)], float],
    interval: Tuple[(float | int)],
    delta_x: (float | int),
) -> float | int:
    """
    Takes function f, an interval (a,b) defined as a tuple, and a partition number n
    and approximates the integral of f on [a,b]

    Parameters:
    - f: function
    - interval: interval (a,b) where b>a
    - n : partition number, how many rectangles to approximate integral

    Returns:
    integral, the result of integration process, "area under the curve".

    Example:
    """
    fvec = np.vectorize(f)
    return np.sum(
        fvec(np.arange(start=interval[0], stop=interval[1], step=delta_x)) * delta_x
    )


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    dtype = MPI.FLOAT
    np_dtype = dtlib.to_numpy_dtype(dtype)
    itemsize = dtype.Get_size()

    if rank == 0:
        # Define function on root process
        function_string = create_function_string_from_user_input()
        f = create_function_on_process(function_string)
        # Check that it was "compiled"
        assert callable(f)

        # User input for integral
        print(
            """Enter start of interval,\
 end of interval, and partition number: """
        )
        a, b, n = input().split()

        # Type convert into appropriate format
        a, b = tuple(map(float, [a, b]))
        n = int(n)

        # Check that user inputs fulfill conditions
        try:
            if n % size != 0:
                raise ValueError(
                    "Number of partitions \
isn't evenly split between processes"
                )
            if a >= b:
                raise ValueError(
                    "The lower limit of the interval\
greater or equal to upper"
                )
        except ValueError as e:
            print(e)

    else:
        function_string = a = b = n = None

        # Evenly partition the interval into subintervals
        # of size partition space, one per process

    comm.Barrier()

    function_string, a, b, n = comm.bcast((function_string, a, b, n), root=0)
    if rank != 0:
        f = create_function_on_process(function_string)

    partition_size = (b - a) / size
    partition_delta = n // size
    n = partition_size / partition_delta
    local_interval = (
        a + rank * partition_size,
        a + (rank + 1) * partition_size,
    )

    local_sum = np.array([riemann_sum_left(f, local_interval, n)], dtype=np_dtype)
    global_sum = np.empty(1, dtype=np_dtype)
    win = MPI.Win.Allocate(itemsize, comm=comm)
    win.Lock(rank=0, lock_type=MPI.LOCK_SHARED, assertion=MPI.MODE_NOCHECK)
    target_rank = 0
    win.Accumulate([local_sum, dtype], target_rank=target_rank, op=MPI.SUM)
    win.Unlock(rank=0)
    comm.Barrier()
    win.Lock(rank=0, lock_type=MPI.LOCK_SHARED, assertion=MPI.MODE_NOCHECK)
    win.Get(global_sum, target_rank=0)
    win.Unlock(rank=0)

    if rank == 0:
        print(f"The integral of {function_string} = {global_sum[0]}")

    win.Free()


if __name__ == "__main__":
    main()
