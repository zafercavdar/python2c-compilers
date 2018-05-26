from cymatrix import multiply_matrices
from time import time
import numpy as np


for x in range(1, 18):
    N = x * 100
    mat = np.random.rand(N, N)
    s = time()
    res = multiply_matrices(mat, mat)
    f = time()
    print("cython; N: {}, time: {}".format(N, f - s))
