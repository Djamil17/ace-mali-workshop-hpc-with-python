import numpy as np


def mag(x: (int | float), y: (int | float)) -> int | float:
    return (x**2 + y**2) ** (1 / 2)


docstring = """"
Calculate the magntiude between two elements element by element in array

    This function two arrays, finds the magnitude of 1rst element and 1rst
    element of second array , 2nd element and 2nd element of the second array
    etc.

    Parameters:
    x (np.ndarray[(int | float)]): The first array
    y (np.ndarray[(int | float)]): The second array

    Returns:
    np.ndarray: the array with each magnitude calculated


"""

mag_vect = np.vectorize(mag, doc=docstring)
x = y = np.arange(10)
x = mag_vect(x, 1)
print(mag_vect.__doc__)
z = mag_vect(x, y)
type(z[0])

mag_vect = np.vectorize(mag, doc=docstring, cache=True, signature="(m),pyth() -> (m)")
x = mag_vect(x, 1)
x = mag_vect(x, y)
print(x)
