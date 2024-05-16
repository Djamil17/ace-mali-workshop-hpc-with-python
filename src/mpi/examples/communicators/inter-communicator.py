from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
status = MPI.Status()

local_leader = 0
remote_leader = 0

if size < 4:
    print("This program needs at least 4 processes to run.")
    comm.Abort(1)

if rank < size // 2:
    color = 0
    key = rank
else:
    color = 1
    key = rank

sub_comm = comm.Split(color, key)


# Create an intercommunicator
intercomm = comm.Create_intercomm(local_leader, sub_comm, remote_leader)

local_rank = sub_comm.Get_rank()
local_size = sub_comm.Get_size()
for dest_rank in range(local_size):
    if dest_rank != local_rank:
        local_message = f"from process {dest_rank}"
        sub_comm.send(local_message, dest=dest_rank)

for src_rank in range(local_size):
    if src_rank != local_rank:
        received_message = sub_comm.recv(source=src_rank)
        print(f"message received at {src_rank}", received_message, end="\n")

intercomm.Free()
sub_comm.Free()
MPI.Finalize()

# Perform communication within each subgroup if needed
# global_data = intercomm.gather(local_data, root=0)
# print(global_data)
