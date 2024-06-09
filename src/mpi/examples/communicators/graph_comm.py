"""
    Demonstrates the creation of a graph topology using MPI with mpi4py.

    This function initializes the MPI environment, creates a graph topology
    with predefined neighbor relationships, and prints the neighbors of each
    process. The program assumes there are exactly 3 processes, each with a
    specific set of neighbors.

    Steps:
    1. Initialize the MPI environment and obtain the rank and size of the communicator.
    2. Ensure the number of processes is exactly 3.
    3. Define the index and edges arrays to specify the graph topology.
    4. Create a graph communicator based on the index and edges.
    5. Retrieve and print the neighbors for each process.
    6. Free the graph communicator.

    This example showcases the use of MPI's graph topology capabilities to define
    and manage custom communication patterns in parallel applications.

    Date: 05/30/2024
    Author: Djamil Lakhdar-Hamina
"""
from mpi4py import MPI


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    assert size == 3
    index = [
        1,
        3,
        4,
    ]  # Process 0 has 1 neighbor, process 1 has 2 neighbors, process 2 has 1 neighbors
    edges = [
        1,
        0,
        2,
        1,
    ]  # Process 0's neighbors is 1 , Process 1's neighbors are 0 and 2, Process 2 is 1

    # Create the graph topology
    graph_comm = comm.Create_graph(index=index, edges=edges)

    # Get the neighbors of the current process in the graph topology
    my_neighbors = graph_comm.Get_neighbors(rank)

    print("Rank {}: My neighbors are {}".format(rank, my_neighbors))

    # Free the graph communicator
    graph_comm.Free()


if __name__ == "__main__":
    main()
