from timeit import timeit

import numpy as np

_ = np.copy()

val1 = timeit(
    "[5*i for i in array_list]",
    setup="import numpy as np; array = np.arange(1e5); \
              array_list=array.tolist()",
    number=1000,
)
val2 = timeit(
    "5*array", setup="import numpy as np; array = np.arange(1e5)", number=1000
)

print(f"element-by-element took {val1} seconds")
print(f"vectorized operation took {val2} seconds")
