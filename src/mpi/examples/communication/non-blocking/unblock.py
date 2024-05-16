# import numpy as np


# def reconstruct_matrix(blocks: np.array) -> np.array:
#     """
#     Reconstructs the original matrix from block partitions.

#     Parameters:
#     - blocks: np.array containing block partitions

#     Returns:
#     np.array of the original matrix

#     """
#     num_blocks = blocks.shape[0]
#     block_size = blocks.shape[1]

#     return matrix

#     # return matrix


# A = np.array(
#     [
#         [[0.0, 0.0], [1.0, 1.0]],
#         [[2.0, 2.0], [3.0, 3.0]],
#         [[0.0, 0.0], [1.0, 1.0]],
#         [[2.0, 2.0], [3.0, 3.0]],
#     ]
# )

# print(reconstruct_matrix(A))

# # 0 0 0 -> 0 0
# # 0 0 1 -> 0 1
# # 0 1 0 -> 1 0
# # 0 1 1 -> 1 1
