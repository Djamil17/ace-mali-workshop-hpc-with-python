import numpy as np

x = np.linspace(0, 1, 5)
x
x_view = x[1:]
x_view[:] = 1.0
x
x_copy = x.copy()
x_copy[1:] = 3.0
x
x_copy
