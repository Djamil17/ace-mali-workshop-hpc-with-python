import time

A = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
x = [1, 2, 3, 4]
y = [0, 0, 0, 0]


def timeit(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        return (t2 - t1), res, func.__name__

    return wrapper


@timeit
def matrix_vector_row_mult(A, b):
    row = len(A)
    column = len(A[0])
    c = [[0] * row for _ in range(0, column)]

    for i in range(0, row):
        for j in range(0, column):
            c = A[i][j] * b[i]
    return c


@timeit
def matrix_vector_col_mult(A, b):
    row = len(A)
    column = len(A[0])
    c = [[0] * row for _ in range(0, column)]
    for j in range(0, row):
        for i in range(0, column):
            c = A[i][j] * b[i]
    return c


for times in [matrix_vector_row_mult(A, x), matrix_vector_row_mult(A, x)]:
    print(f"operation took {times[0]}")
