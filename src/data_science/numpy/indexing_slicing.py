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

# initialize 3 matrices each of 4 rows and 2 columns
x = np.random.randint(0, 5, (4, 4, 3))
x
# pick out second matrix , 3rd row, first column
x[1, 2, 0]

# after the second matrix, pick out after 3nd row , reverse order of columns
x[1:, 2:, ::-1]
