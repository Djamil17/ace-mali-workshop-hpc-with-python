import numpy as np

MEANING_OF_LIFE = 42
rng = np.random.RandomState(MEANING_OF_LIFE)
x = rng.randint(10, size=(3, 4))
x
x == 6

np.sum(x < 6)

np.any(x == 6)
np.all(x == 6)

x_subarray = x[x < 6]

# between 6 inclusive and 8 exclusive
x[(x >= 6) & (x < 8)]
