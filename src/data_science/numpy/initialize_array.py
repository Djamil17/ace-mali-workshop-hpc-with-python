import numpy as np

np.array([1, 2, 3, 4, 5], dtype="float")
np.array([1, 2, 3, 4, 5], dtype="f")
np.array([1, 2, 3, 4, 5], dtype="int")
np.array([1, 2, 3, 4, 5], dtype="i")

np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype="<f8")

# array of zeros created from scatch
np.zeros(5, dtype="int")

# matrix of ones created from scatch
np.ones((3, 2), dtype="int")

# array of value specified created from scatch
np.full((2, 2), 42, dtype="int")

# array of linear sequence
begin = 10
end = 20
step = 2
np.arange(begin, end, step, dtype="int")

# 5 values spacd between 1
np.linspace(0, 1, 5)

np.random.random((5, 5))

# array of random values based on n dimensions
np.random.random((3, 4))

# array of random values in an interval 0-10 and (,,,) dimensions
np.random.randint(0, 10, (3, 3))

message = b"ace is the best!"
np.frombuffer(message, dtype="|S1", count=5)
