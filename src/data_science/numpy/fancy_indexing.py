import numpy as np

rand = np.random.RandomState(1917)
x = rand.randint(10, size=10)
x
indexes = [1, 4, 5]

x[indexes]

X = np.arange(12).reshape(3, 4)
index_row = np.array([0, 1, 2])
index_col = np.array([1, 2, 1])
X
X[index_row, index_col]


X[index_row, np.newaxis, index_col]
