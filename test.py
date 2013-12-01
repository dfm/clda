import numpy as np
from ctr import _cf

U = np.random.rand(5, 2)
V = np.random.rand(10, 2)

print(V)

user_items = [[4, 5], [3], [6, 8], [6, 5, 8], [1, 0, 5]]

_cf.update(U, V, user_items)
