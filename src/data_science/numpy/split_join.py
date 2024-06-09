import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([6, 7, 8, 9, 10])

np.concatenate([x, y])
np.concatenate((x, y))

z = np.array([1, 2, 3])
np.concatenate((x, y, z))

x = np.array([[1, 2, 3], [1, 2, 3]])
y = np.array([[4, 5, 6], [7, 8, 9]])
# create a new matrix by adding rows of y to x
np.concatenate([x, y], axis=0)
# create a new matrix by adding colunmns of y to x
np.concatenate([x, y], axis=1)

np.hstack([x, y])
np.vstack([x, y])

x = np.array([[1, 2, 3], [1, 2, 3]])
y = np.array([1, 3, 4])

np.vstack((x, y))

x = np.arange(10)
np.split(x, [2, 7])
x = np.arange(25).reshape((5, 5))
np.split(x, [2, 3], axis=0)
