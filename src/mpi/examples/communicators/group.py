from mpi4py import MPI

# 0 1 2
# 3 4 5

comm = MPI.COMM_WORLD
# first row ranks to add to new group
process_ranks = [0, 1, 2]

group_world = comm.Get_group()

# create new group
first_row_group = group_world.Incl(process_ranks)

first_row_comm = comm.Create(first_row_group)

print(first_row_comm)
