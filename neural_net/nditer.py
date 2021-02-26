import numpy as np

x = np.array([[1,2], [3,4]])
it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
print(x)
while not it.finished:
    idx = it.multi_index
    print(idx, x[idx], it[0])
    it.iternext()