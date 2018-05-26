from time import time
from concurrent.futures import ProcessPoolExecutor


def op(x):
    return 2 * x - 1


def iterable(size):
    return [i for i in range(0, size)]


def multithread_apply(func, iterable, max_workers, chunksize):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tick = time()
        executor.map(func, iterable, chunksize=chunksize)
        tock = time()
        print("with {} max workers and chunksize of {}: {}".format(max_workers, chunksize,
                                                                   tock - tick))


def test_chunksize_effect():
    N = 1000000
    p = 4
    s = time()
    ds = iterable(N)
    list(map(op, ds))
    f = time()
    print("single thread: {}".format(f - s))

    multithread_apply(op, ds, p, 250000)


def test_multithread_n_workers():
    iterables = iterable(1000000)
    for n_worker in range(1, 100):
        multithread_apply(op, iterables, n_worker, 100)


def test_multithread_chunksize():
    for x in range(1, 60):
        multithread_apply(op, iterable, 4, x * 5)
