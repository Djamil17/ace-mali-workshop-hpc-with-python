import numpy as np

dt = np.dtype("<f8")
print(dt)
print(dt.byteorder)
print(dt.itemsize)

point_type = np.dtype(
    [
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
        ("info", np.float64, (3,)),
    ]
)


points = np.array(
    [(1.0, 2.0, 3.0, (1.0, 2.0, 3.0)), (4.0, 5.0, 6.0, (7.0, 8.0, 9.0))],
    dtype=point_type,
)

print(points[1])(4.0, 5.0, 6.0, [7.0, 8.0, 9.0])
