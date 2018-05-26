from multiprocessing import Pool
from time import time


def iterable(size):
    return [i for i in range(0, size)]


def fib(x):
    fibs = [1, 1]
    for _ in range(0, 20):
        new = fibs[-1] + fibs[-2]
        fibs.append(new)
    return fibs[-1]


def multiprocessing_apply(func, iterable, cores):
    tick = time()
    with Pool(cores) as p:
        p.map(func, iterable)
    tock = time()
    print("with {} cores : {}".format(cores, tock - tick))


def test_multiprocessing():
    ds = iterable(10000000)
    for c in range(1, 21):
        multiprocessing_apply(fib, ds, c)
