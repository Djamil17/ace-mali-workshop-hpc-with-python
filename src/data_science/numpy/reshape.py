import numpy as np

x = np.linspace(0, 1, 9)
x
x_reshaped = x.reshape((3, 3))

x = np.random.rand(9, 9)
x.reshape((3, 3, 3))

x = np.linspace(0, 1, 9)

x.reshape((9, 1))

x.reshape((1, 9))
