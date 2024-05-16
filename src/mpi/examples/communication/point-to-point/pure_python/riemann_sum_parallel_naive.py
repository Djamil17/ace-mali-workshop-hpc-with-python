"""
Perform a left-sided riemann sum given user-inputed function, interval,
and a partition number in a parallel fashion making use of send and receive.

Date: 05/14/2024
Author: Djamil Lakhdar-Hamina
Last Modified : 05/15/2024

"""

from itertools import islice
from typing import Callable, Generator, Iterable, List, Type

from mpi4py import MPI


def batched(iterable: Iterable, iterable_type: Type, n: (int | float)) -> Generator:
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
    f: Callable[[float, int], float],
    x: List[(float | int)],
    delta_x: List[(float | int)],
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
    sum = 0
    for x_i in x:
        sum += f(x_i) * delta_x
    return sum


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    global_sum = 0

    # make sure local copies on each processor
    f = function_string = a = b = n = delta_x = local_interval = None

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

        # Evenly partition the interval into subintervals
        # of size partition space, one per process
        delta_x = (b - a) / n
        x = [a + i * delta_x for i in range(0, n)]
        partition_size = n // size
        batched_x = list(batched(x, list, partition_size))
        print(batched_x)

    comm.Barrier()

    # Distribute the info and sub-interval across processes.
    # Receive the info and compile function
    if rank == 0:
        local_interval = batched_x[0]
        local_sum = riemann_sum_left(f, local_interval, delta_x)
        for i in range(1, size):
            info = (batched_x[i], function_string, delta_x)
            comm.send(info, dest=i)
    else:
        local_interval, function_string, delta_x = comm.recv(source=0)
        f = create_function_on_process(function_string)

    # Perform the local summation then send back
    if rank != 0:
        local_sum = riemann_sum_left(f, local_interval, delta_x)
        comm.send(local_sum, dest=0)
    else:
        global_sum = local_sum
        for process in range(1, size):
            local_sum = comm.recv(source=process)
            global_sum += local_sum
        print(f"sum: {global_sum}")

    return global_sum


if __name__ == "__main__":
    main()
