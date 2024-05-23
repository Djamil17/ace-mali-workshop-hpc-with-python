"""
Generate random RNA file (aguc) of length N. Read in this
file in parallel fashion and create array of chars. Then
count each letter per process. Sum total and write to new file

Date: 05/20/2024
Author: Djamil Lakhdar-Hamina

"""

import numpy as np
from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # size = comm.Get_size()

    amode = MPI.MODE_CREATE | MPI.MODE_RDWR | MPI.MODE_APPEND

    fh = MPI.File.Open(comm, "./data.rna", amode)
    nucleotides = ["a", "g", "u", "c"]
    buffer = np.random.choice(nucleotides, 10)
    chunk_size = buffer.nbytes
    offset = chunk_size * rank
    fh.Write_at_all(offset, buffer)
