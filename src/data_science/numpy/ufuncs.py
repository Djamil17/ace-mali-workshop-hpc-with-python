import numpy as np

x = np.arange(10)
# unary ufunc negation
-x
# binary ufunct addition
np.add(x, x)
x + x
# overload example
np.absolute(-x)
abs(-x)
