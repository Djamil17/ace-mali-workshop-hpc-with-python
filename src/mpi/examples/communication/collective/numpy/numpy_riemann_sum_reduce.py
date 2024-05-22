"""
Implement left-handed riemann sum using NumPy.

Date: 05/22/2024
Author: Djamil Lakhdar-Hamina

"""

from typing import Callable, Type

import numpy as np
from mpi4py import MPI


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
    f: Callable[[float, int], float], interval: tuple, n: int, dtype: Type = float
) -> np.ndarray:
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
    a = interval[0]
    b = interval[1]
    delta_x = (b - a) / n
    x = np.arange(a, b, delta_x)
    integral = sum(f(x) * delta_x)
    return np.array(integral, dtype=dtype)


def main():
    root = 0
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    global_sum = np.empty(1, dtype=float)
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
        info = np.array([a, b, n], dtype=float)

        # Check that user inputs fulfill conditions
        try:
            if info[2] % size != 0:
                raise ValueError(
                    "Number of partitions \
isn't evenly split between processes"
                )
            if info[0] >= info[1]:
                raise ValueError(
                    "The lower limit of the interval\
 greater or equal to upper"
                )
        except ValueError as e:
            print(e)
    else:
        info = np.empty(3, dtype=float)
        function_string = None

    comm.Bcast([info, MPI.FLOAT], root=0)
    function_string = comm.bcast(function_string, root=0)
    a, b, n = info
    if rank != 0:
        f = create_function_on_process(function_string)

    # use rank number to create a interval in tuple form
    partition_size = (b - a) / size
    partition_number = n // size
    local_interval = (
        a + rank * partition_size,
        a + (rank + 1) * partition_size,
    )
    local_sum = riemann_sum_left(f, local_interval, partition_number)
    # we use a double to allow for greater precision otherwise overflow can happen
    comm.Reduce(
        [local_sum, MPI.DOUBLE], [global_sum, MPI.DOUBLE], op=MPI.SUM, root=root
    )

    if rank == 0:
        print("the global sum is:", global_sum[0])

    return global_sum


main()
