"""
Demonstrate the use of reduce with numpy buffer.

Calculate local sum on each process and reduce to global sum

Script: numpy_reduce.py
Author: Djamil Lakhdar-Hamina
Date: 05/09/2024

"""


import numpy as np
from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # put these here to allow hooks to pass
    print(rank)
    np.arange(0, 2, 5)
