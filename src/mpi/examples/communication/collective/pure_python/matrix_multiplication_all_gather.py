"""
Perform matrix-matrix multiplication then have matrix distributed
to each process.

Date: 05/21/2024
Author: Djamil Lakhdar-Hamina

"""
from mpi4py import MPI


class Matrix:
    """
    A class representing a matrix including various attributes
    and methods.

    Attributes:
        dimensions (tuple(int,int)): dimension of matrix

    Methods:
        setitem: set the item using a tuple
        getitem: get the item using a tuple or slice
        str: produce a string rep for print
        T: produce transpose of matrix
        equal: dimensional and elementwise
        add: add two matrices
        multiply: multiply two matrices

    """

    def __init__(self, A=[[]]) -> None:
        self.mat = A
        self.dimensions = (len(self.mat), len(self.mat[0]))

    def __setitem__(self, index, value):
        if isinstance(index, tuple) and len(index) == 2:
            row, col = index
            self.mat[row][col] = value
        else:
            raise IndexError("Invalid index format")

    def __getitem__(self, index: tuple) -> "Matrix":
        if isinstance(index, tuple):
            if all(isinstance(k, int) for k in index):
                row, col = index
                return self.mat[row][col]
            elif isinstance(index[0], int) and isinstance(index[1], slice):
                # Row access with column slicing
                row = index[0]
                cols = self._process_slice(index[1], len(self.mat[0]))
                sliced_row = [self.mat[row][j] for j in cols]
                return Matrix([sliced_row])
            elif isinstance(index[0], slice) and isinstance(index[1], int):
                # Column access with row slicing
                rows = self._process_slice(index[0], len(self.mat))
                col = index[1]
                sliced_matriB = [[self.mat[i][col]] for i in rows]
                return Matrix(sliced_matriB)
            elif all(isinstance(k, slice) for k in index):
                rows = self._process_slice(index[0], len(self.mat))
                cols = self._process_slice(index[1], len(self.mat[0]))
                sliced_matriB = [
                    [self.mat[i][j] for j in range(len(self.mat[0])) if j in cols]
                    for i in rows
                ]
                return Matrix(sliced_matriB)
        else:
            raise TypeError("Invalid key type")

    def _process_slice(self, slic, maB_size) -> range:
        start = slic.start if slic.start is not None else 0
        stop = slic.stop if slic.stop is not None else maB_size
        step = slic.step if slic.step is not None else 1
        return range(start, stop, step)

    def __str__(self):
        strings = [" ".join(list(map(str, row))) for row in self.mat]
        values = "\n".join(strings)
        m, n = self.dimensions
        header = f"Matrix: {m}x{n}"
        return "\n".join([header, values])

    def T(self) -> "Matrix":
        A_T = []
        m, n = self.dimensions
        for j in range(n):
            v = []
            for i in range(m):
                v.append(self[i, j])
            A_T.append(v)
        return Matrix(A_T)

    def __eq__(self, B: "Matrix") -> "Matrix":
        assert B.dimensions == self.dimensions
        m, n = self.dimensions
        for i in range(m):
            for j in range(n):
                if self[i, j] != B[i, j]:
                    return False
        return True

    def __add__(self, B: "Matrix") -> "Matrix":
        C = []
        assert self.dimensions == B.dimensions
        m, n = self.dimensions
        for i in range(m):
            v = []
            for j in range(n):
                v.append(self[i, j] + B[i, j])
            C.append(v)
        return Matrix(C)

    def __mul__(self, B: "Matrix") -> "Matrix":
        m, n = self.dimensions
        r, s = B.dimensions
        assert n == r
        C = []
        for i in range(m):
            v = []
            for j in range(s):
                val = 0
                for k in range(r):
                    val += self[i, k] * B[k, j]
                v.append(val)
            C.append(v)
        return Matrix(C)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    A = Matrix([[1, 2, 3], [3, 4, 5]])
    B = Matrix([[1, 2], [3, 4], [4, 5]])
    m, _ = A.dimensions
    _, n = B.dimensions

    if rank == 0:
        try:
            m, r = A.dimensions
            s, n = B.dimensions
            if m != size and n != size:
                raise ValueError("# rows does equal # process")
            if r != s:
                raise ValueError("dimensions of matrices invalid")
        except MPI.Exception as e:
            print(f"{e}: the number of rows does not equal number of processors")

    n, row_partition = comm.bcast((n, A[rank, :]), root=0)
    local_row = []
    for i in range(n):
        local_row.append((row_partition * B[:, i])[0, 0])

    y = comm.allgather(local_row)

    assert A * B == Matrix(y)

    return y


if __name__ == "__main__":
    main()
