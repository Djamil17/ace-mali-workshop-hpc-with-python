from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    clone = comm.Clone()

    predefined_attrs = MPI.Comm.Get_attr(MPI.COMM_WORLD, MPI.TAG_UB)
    print(predefined_attrs)

    key = clone.Create_keyval()
    clone.Set_attr(key, "foobar")
    attr_value = clone.Get_attr(key)
    print("Attribute value:", attr_value)
    clone.Free_keyval(key)
    # # Print the names and values of predefined attributes
    # print("Predefined attributes of the communicator:")
    # for attr in predefined_attrs:
    #     value = comm.Get_attr(attr)
    #     print(f"{attr}: {value}")


main()
