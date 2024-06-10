"""
Exhibit the use of Numba and NumPy to create and use a universal function (ufunc).

This program defines a ufunc to calculate the magnitude,
optimized with Numba for performance. It then applies this ufunc to arrays of points.

Functions:
    magnitude(x: float, y: float) -> float: Computes the magnitude
    between points
  main() -> None: Initializes arrays of points and prints their magnitude.
"""
import numba as nb
import numpy as np


@nb.vectorize([nb.float32(nb.float32, nb.float32)])
def magnitude(x: float, y: float) -> float:
    return (x**2 + y**2) ** (1 / 2)


def main() -> None:
    points_x, points_y = (
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
    )

    print(f"magnitudes : {magnitude(points_x, points_y)}")


if __name__ == "__main__":
    main()
