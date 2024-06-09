import numpy as np

a = np.arange(3)
M = np.ones((3, 3))
print(a + 5)
print(M + a)

a = np.arange(3)
b = np.arange(3)[:, np.newaxis]
print(a, "\n")
print(b, "\n")
print(a + b)

M = np.ones((2, 3))
a = np.arange(3)
print("M.shape =", M.shape)
print("a.shape =", a.shape)
# By rule 1 , if two arrays differ in number of dimensions ,
# the shape of the one with fewer dimensions is padded with ones on
# its leading left side.
print("M.shape ->", M.shape)
print("a.shape->", "(1,3)")

print("M.shape =", M.shape)
print("a.shape ->", "(2,3)")
a = np.arange(3).reshape((3, 1))
b = np.arange(3)
print("a.shape =", (3, 1))
print("b.shape =", (3,))
