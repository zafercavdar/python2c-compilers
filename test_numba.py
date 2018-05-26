from numba import jit
from numba.decorators import autojit
from time import time
from math import log10
import re
import numpy as np

regex = r"\s+"


def op(x):
    return 2 * x - 1


def split_by_pattern(x):
    return re.split(regex, x)


def read_file(path):
    f = open(path, "r")
    return [line for line in f]


def iterable(size):
    return [i for i in range(0, size)]


def numba_apply(func, iterable):
    def pure_python_call():
        return [func(x) for x in iterable]

    @jit
    def numba_call():
        return [func(x) for x in iterable]

    s = time()
    pure_python_call()
    f = time()
    py = f - s
    s = time()
    numba_call()
    f = time()
    nb = f - s
    print("{} {} {}".format(log10(len(iterable)), py, nb))


def test_numba_vs_python_simple_op():
    for x in range(0, 17):
        size = pow(3, x)
        iterables = iterable(size)
        numba_apply(op, iterables)


def test_numba_vs_python_The_Idiot():
    for k in range(1, 50):
        dataset = read_file("The_Idiot.txt") * k
        numba_apply(split_by_pattern, dataset)


def cos_sim_matrix(arr):
    N = arr.shape[0]
    dist = np.zeros((N, N))
    for i, row in enumerate(arr):
        for j, col in enumerate(arr):
            dot_product = np.dot(row, col)
            norm_a = np.linalg.norm(row)
            norm_b = np.linalg.norm(col)
            dist[i][j] = dot_product / (norm_a * norm_b)
    return dist


cos_sim_matrix_numba = autojit(cos_sim_matrix)


def matrix_multiplication(m1, m2):
    return m1 * m2


matrix_multiplication_numba = autojit(matrix_multiplication)


def matrix_multiplication_loops(m1, m2):
    M = len(m1)
    N = len(m1[0])
    Q = len(m2[0])
    res = np.zeros((M, Q))
    sum = 0

    for c in range(M):
        for d in range(Q):
            for k in range(N):
                sum += m1[c][k] * m2[k][d]

            res[c][d] = sum
            sum = 0
    return res


matrix_multiplication_loops_numba = autojit(matrix_multiplication_loops)


def matrix_substraction(m1, m2):
    return m1 ** 5


matrix_substraction_numba = autojit(matrix_substraction)


def test_numba_with_numpy():
    # N = 1000
    D = 20

    sizes = [x * 100 for x in range(1, 11)]
    for size in sizes:
        arr = np.random.rand(size, D)
        s = time()
        cos_sim_matrix(arr)
        f = time()
        temp = f - s

        s = time()
        cos_sim_matrix_numba(arr)
        f = time()
        print("numpy: {} numpy+numba: {}".format(temp, f - s))


def test_numba_matrix_multiplication():
    Ns = [x * 100 for x in range(1, 18)]
    compiler = np.random.rand(5, 5)
    matrix_multiplication_numba(compiler, compiler)

    for N in Ns:
        m = np.matrix(np.random.rand(N, N))
        s = time()
        matrix_multiplication(m, m)
        f = time()
        temp = f - s

        s = time()
        matrix_multiplication_numba(m, m)
        f = time()
        print("numpy: {} numpy+numba: {}".format(temp, f - s))


def test_numba_matrix_multiplication_loops():
    Ns = [x * 100 for x in range(1, 18)]
    compiler = np.random.rand(5, 5)
    matrix_multiplication_loops_numba(compiler, compiler)

    for N in Ns:
        m = np.random.rand(N, N)
        s = time()
        matrix_multiplication_loops(m, m)
        f = time()
        temp = f - s

        s = time()
        matrix_multiplication_loops_numba(m, m)
        f = time()
        print("python-loops: {} python-loops+numba: {}".format(temp, f - s))


def test_numba_matrix_substraction():
    sizes = [x * 500 for x in range(1, 21)]
    for size in sizes:
        m = np.matrix(np.random.rand(size, size))
        s = time()
        matrix_substraction(m, m)
        f = time()
        temp = f - s

        s = time()
        matrix_substraction_numba(m, m)
        f = time()
        print("numpy: {} numpy+numba: {}".format(temp, f - s))


test_numba_matrix_multiplication_loops()
