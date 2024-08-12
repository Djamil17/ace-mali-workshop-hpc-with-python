import numpy as np

x = np.arange(1, 101, dtype="float32")
np.add.reduce(x)
np.multiply.reduce(x)
x = np.float64(x)
np.multiply.reduce(x)


x = np.arange(1, 10, dtype="float64")

np.multiply.accumulate(x)

x = np.arange(25).reshape((5, 5))

np.mean(x)
x.mean()

# by row
x.mean(axis=0)

# by column
x.mean(axis=1)
