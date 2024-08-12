"""
This program demonstrates the use of a decorator to print the result of a function.
It includes a bubble sort implementation that is decorated to print its result.

The main steps are:
1. Define a simple print function to print the result of a given function.
2. Define a decorator that wraps a function to print its result when called.
3. Implement the bubble sort algorithm and decorate it to print its sorted list.
4. Execute the bubble sort function on an example list.

Usage:
------
Simply run the script to see the output of the bubble sort function.

"""
from functools import wraps


def print_result(func):
    """
    Print the result of the given function.

    Parameters:
    func (Callable): The function to execute and print its result.
    """
    result = func
    print(result)


def decorator_print_result(func, on=True):
    """
    Decorator that prints the result of the decorated function.

    Parameters:
    func (Callable): The function to wrap.

    Returns:
    Callable: The wrapped function that prints its result.
    """

    @wraps(func)
    def printer(*args, **kwargs):
        result = func(*args, **kwargs)
        print(result)
        return result

    return printer


@decorator_print_result
def bubble_sort(x: list, in_place=False, out=None) -> list:
    """
    Sort a list using the bubble sort algorithm.

    Parameters:
    x (list): The list to sort.
    in_place (bool, optional): If True, sort the list in place. Default is False.
    out (list, optional): If provided, the sorted list will be stored in this parameter.

    Returns:
    list: The sorted list.
    """
    n = len(x)
    for i in range(n):
        for j in range(i, n):
            if x[j - 1] > x[j]:
                swap = x[j]
                x[j] = x[j - 1]
                x[j - 1] = swap
    return x


if __name__ == "__main__":
    example = [3, 3, 6, 7]
    print_result(bubble_sort(example))
    bubble_sort(example)
