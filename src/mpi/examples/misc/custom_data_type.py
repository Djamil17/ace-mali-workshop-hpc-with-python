from itertools import islice

import numpy as np
from mpi4py import MPI


def batched(iterable, bundle_type, n=2):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := bundle_type(islice(it, n)):
        yield batch


def definetype(field_names: list, field_dtypes: list, size: int) -> MPI.Datatype:
    num = 2
    dtypes = list(zip(field_names, field_dtypes))
    a = np.zeros(num, dtype=dtypes)

    struct_size = a.nbytes // num
    offsets = [a.dtype.fields[field][1] for field in field_names]

    mpitype_dict = {np.float64: MPI.DOUBLE}  # etc
    field_mpitypes = [mpitype_dict[dtype] for dtype in field_dtypes]

    structtype = MPI.Datatype.Create_struct(
        [1] * len(field_names), offsets, field_mpitypes
    )
    structtype = structtype.Create_resized(0, struct_size)
    structtype.Commit()
    return structtype


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Define custom data type
    struct_field_names = ["x", "y"]
    struct_field_types = [np.float64, np.float64]
    point_type = list(zip(struct_field_names, struct_field_types))
    custom_type = definetype(struct_field_names, struct_field_types, 2)
    buf = np.zeros(size, dtype=point_type)

    if rank != 0:
        comm.Recv([buf, custom_type], source=0, tag=0)
        print(buf)
    else:
        buf = np.array(list(batched(range(0, size * 2), tuple)), dtype=point_type)
        for p in range(1, size):
            comm.Send([buf, custom_type], dest=p, tag=0)


main()
