"""
Depict some attribute info about ndarray for 2 dimensional
example.

Date: 05/21/2024
Author: Djamil Lakhdar-Hamina

"""

import numpy as np


def main():
    x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype=">i4", order="c")
    x.flags.writeable = False
    print(f"1. number of bytes: {x.nbytes}")
    print(f"2. number of dimensions: {x.ndim}")
    print(f"2.1 shape of array: {x.shape}")
    print(f"3. datatype and itemsize: {x.dtype, x.dtype.itemsize}")
    print(f"4. seperation for dimensions: {x.strides}")
    print(
        f"5. byte order ('=' means native, < means little-endian, > means big-endian)\
: {x.dtype.byteorder}"
    )
    print(f"6-7.: {x.flags}")


if __name__ == "__main__":
    main()
