"""
This MPI program demonstrates the creation and use of intercommunicators
to facilitate communication between two subgroups of processes. The program
is designed for a minimum of 4 processes and splits the processes into two
subgroups. Each subgroup has a local leader that communicates with its
children within the subgroup.

The steps performed are:
1. Initialize MPI and obtain rank and size.
2. Split the processes into two subgroups based on rank.
3. Create an intercommunicator between the two subgroups.
4. Perform intra-group communication: local leader sends messages to its
   children within the subgroup.
5. Free the intercommunicators and sub-communicators.
6. Finalize MPI.

Run the program with at least 4 processes to observe the communication
patterns.

Functions:
- main: Main function to execute the MPI communication.
"""
from mpi4py import MPI


# Initialize MPI
def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_leader = 0
    remote_leader = 0
    assert size >= 4

    if rank < size // 2:
        color = 0
        key = rank
    else:
        color = 1
        key = rank

    sub_comm = comm.Split(color, key)

    # Create an intercommunicator
    if color == 0:
        intercomm = comm.Create_intercomm(local_leader, sub_comm, remote_leader, 0)
    else:
        intercomm = comm.Create_intercomm(
            local_leader, sub_comm, remote_leader, size // 2
        )

    local_rank = sub_comm.Get_rank()

    if color == 0:
        if local_rank == local_leader:
            for p in range(1, size // 2):
                intercomm.send(None, dest=p)
                print(
                    f"rank:local_rank : sending from {rank}:{local_rank} \
to {p}:{local_rank}..."
                )
        else:
            intercomm.recv(source=local_leader)
            print(
                f"rank:local_rank : {rank}:{local_rank} \
received from {local_leader}:{local_leader}",
                end="\n",
            )
    else:
        if local_rank == local_leader:
            for p in range(size // 2 + 1, size):
                intercomm.send(None, dest=p)
                print(
                    f"rank:local_rank : sending from {rank}:{local_rank}\
 to {p}:{local_rank}..."
                )
        else:
            intercomm.recv(source=size // 2)
            print(
                f"rank:local_rank : {rank}:{local_rank} \
received from {size//2}:{local_leader}",
                end="\n",
            )

    intercomm.Free()
    sub_comm.Free()
    MPI.Finalize()


if __name__ == "__main__":
    main()
