from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


index = [
    1,
    3,
    4,
]  # Process 0 has 1 neighbor, process 1 has 2 neighbors, process 2 has 2 neighbors
edges = [
    1,
    0,
    2,
    1,
]  # Process 0's neighbors is 1 , Process 1's neighbors are 0 and 2, Process 2's is 1

# Create the graph topology
graph_comm = comm.Create_graph(index=index, edges=edges)

# Get the neighbors of the current process in the graph topology
my_neighbors = graph_comm.Get_neighbors(rank)

print("Rank {}: My neighbors are {}".format(rank, my_neighbors))

# Free the graph communicator
graph_comm.Free()
