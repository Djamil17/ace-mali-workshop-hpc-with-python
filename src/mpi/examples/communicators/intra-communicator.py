from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Split the communicator into two sub-communicators
if rank < size // 2:
    color = 0
    key = rank
else:
    color = 1
    key = rank


# put into one of two groups if data defined in one or the other is 0 or 1
sub_comm = comm.Split(color, key)

# Perform communication within the sub-communicator
sub_rank = sub_comm.Get_rank()
sub_size = sub_comm.Get_size()

print(f"Rank {rank}: Sub-rank {sub_rank} of size {sub_size}")

sub_comm.Free()
