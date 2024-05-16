from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# want to create a 2x2 square grid with periodicity
n = 2
cart_comm = comm.Create_cart(dims=[n, n], periods=[True, True], reorder=True)

print(f"rank {rank} has coordinates {cart_comm.Get_coords(rank)}")

# Perform communication within the Cartesian communicator
# (e.g., shift communication)
left_neighbor, right_neighbor = cart_comm.Shift(1, 1)
print(f"Rank {rank}: Left neighbor {left_neighbor}, Right neighbor {right_neighbor}")

up_neighbor, down_neighbor = cart_comm.Shift(0, 1)
print(f"Rank {rank}: Up neighbor {up_neighbor}, Down neighbor {down_neighbor}")

sub_cart_comm = cart_comm.Sub([False, True])

print(f"Rank {rank}: has sub-coordinates {sub_cart_comm.coords}")

# Free the Cartesian communicators
sub_cart_comm.Free()
cart_comm.Free()
