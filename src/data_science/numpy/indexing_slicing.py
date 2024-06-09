import numpy as np

x = np.random.randint(0, 10, (3,))
x
x[1]
x[-2]

x = np.random.randint(0, 10, (3, 3, 2))
x
x[1, 1, 1]

x = np.random.randint(0, 10, (10,))
x
# after the second
x[2:]
# from the 4th to the 9th element not including 9
x[3:8]

x = np.random.randint(0, 5, (3, 3, 2))
# after the second matrix
x[1:, 2:, 0]
