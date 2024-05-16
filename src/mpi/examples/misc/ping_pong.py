"""
Send a token between two processors N times where N is user-specified.
Simulates a game of ping-pong.

Date: 05/13/2024
Author: Djamil Lakhdar-Hamina
"""
from mpi4py import MPI


def main():
    token = 0

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    assert size == 2

    rounds = None
    if rank == 0:
        print("input number of rounds:")
        rounds = int(input())

    rounds = comm.bcast(obj=rounds, root=0)

    for round in range(rounds):
        if rank == 0:
            print("round:", round + 1)
            comm.send(token, dest=1)
            token = comm.recv(source=1)
            token += 1
        else:
            token = comm.recv(source=0)
            comm.send(token, dest=0)


if __name__ == "__main__":
    main()
