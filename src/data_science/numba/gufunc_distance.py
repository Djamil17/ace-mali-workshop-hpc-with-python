import numpy as np
from numba import guvectorize


# Define the generalized universal function using guvectorize
@guvectorize(["void(float64[:], float64[:], float64[:])"], "(n),(n)->()", nopython=True)
def compute_distance(point1, point2, result):
    """
    Compute the Euclidean distance between two points in 3D space.

    Parameters:
    point1 : array_like
        First point in 3D space, an array of shape (3,)
    point2 : array_like
        Second point in 3D space, an array of shape (3,)
    result : array_like
        Output array to store the computed distance
    """
    diff = 0.0
    for i in range(point1.shape[0]):
        diff += (point1[i] - point2[i]) ** 2
    result[0] = np.sqrt(diff)


if __name__ == "__main__":
    # Example arrays of 3D points
    points1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    points2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])

    # Compute the distances
    distances = np.empty(points1.shape[0])
    compute_distance(points1, points2, distances)

    print("Distances between points:", distances)
