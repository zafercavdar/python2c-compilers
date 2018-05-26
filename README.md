# Parallelization benchmarks

Spark is slower than we expected since re-distribution of data takes considerable amount of time according to results of pysparkling tests.

After that we decided to run these operations with pure python code in a **not-distributed but parallelized** environment. Python3 provides a multithreading library from `concurrent.futures.ProcessPoolExecutor`.

Firstly, I tested the number of workers (threads)' effect on performance.  
* Input: list of 1 M integers  
* Operation: f(x) = 2*x - 1  
![](https://cdn1.imggmi.com/uploads/2018/5/24/1a5bc60f437acd7cd8ff9a6040317ab6-full.png)

Surprisingly, increasing number of threads didn't increase the performance. I checked a few websites if using threads are correct thing to do or not. This is what I have found:  

> What are threads used for in Python?  
>   * In GUI applications to keep the UI thread responsive  
>   * IO tasks (network IO or filesystem IO)  

>Threads should not be used for CPU bound tasks. Using threads for CPU bound tasks will actually result in worse performance compared to using a single thread. The CPython implementation has a Global Interpreter Lock (GIL) which allows only one thread to be active in the interpreter at once. This means that threads cannot be used for parallel execution of Python code. While parallel CPU computation is not possible, parallel IO operations are possible using threads. This is because performing IO operations releases the GIL.

> For parallel execution of tasks, the **multiprocessing** module can be used.

So, I tested `multiprocessing` module with our basic operation again.
* Input: list of 10 M integers  
* Operation: f(x) = 2*x - 1  

![](https://cdn1.imggmi.com/uploads/2018/5/24/a4779215da70c4497caf885917d43e5c-full.png)

This time increasing number of processes increased the performance ~30%. The reason why we didn't get 100% better performance could be that time required to combine computed results by each core may take more time than total computation time. So, we decided to test `multiprocessing` module with more complex `fibonacci` operation.

![](https://cdn1.imggmi.com/uploads/2018/5/25/726851a77d3e430b524bb71ad68830f7-full.png)

This graph shows the performance improvement by increasing number of processes with `fib(20)` operation. I increased the operation's complexity and did the same benchmark with `fib(40)`. But the resulting graph is similar. Using 2 cores brings 13% more performance for `fib(50)` operation. WHY?

# Numba benchmarks
>  With a few annotations, array-oriented and math-heavy Python code can be just-in-time compiled to native machine instructions, similar in performance to C, C++ and Fortran, without having to switch languages or Python interpreters.

Firstly, I tested performance of `numba` with our basic dataset and operation.  
* Input: python list of integers with varying size
* Operation: f(x)= 2x - 1

![](https://cdn1.imggmi.com/uploads/2018/5/24/2dfb1d41f0e9cdba98fa0501e1b98ce0-full.png)

Due to time required to compile the python code, `numba` does worse than `python` until the dataset size becomes 1M. But `numba` couldn't outperform `pure python` with this python list mapping operation. So, I decided to test its performance with `string` operations.

Secondly, I imported `The_Idiot.txt` line by line.  
* Input: k copies of lines in `The_Idiot.txt` as a python list (k is varying)
* Operation: f(x) = re.split("\s+", line)

![](https://cdn1.imggmi.com/uploads/2018/5/24/836c620aae3398533c4823ab67d4179d-full.png)

Average performance ratio: 0.97. `Numba` and `python` performs very similarly on string operations independent of dataset size.

Thirdly, I tested performance of `Numba` and pure python with `numpy` operations.
* Input: 2D `numpy` array with varying first size dimension, second dimension size=20
* Operation: `cosine similarity` between all rows.

![](https://cdn1.imggmi.com/uploads/2018/5/25/e202e131c409b0108679e83d62757b22-full.png)

Unfortunately, `numba` and `python` do perform similarly again. I checked out some blog posts about when it's suitable to use `numba`. One writer says:
> Numba will be a benefit for functions with the following characteristics:
> * Run time is primarily due to NumPy array element memory access or numerical operations (integer or float) more complex than a single NumPy function call.
> * Functions which work with data types that are frequently converted by NumPy functions to int64 or float64 for calculations (like int8 and int16).
> * The function is called many times during normal execution. Compilation is slow, so if the function is not called more than once, the execution time savings is unlikely to compensate for compilation time. The function execution time is larger than the Numba dispatcher overhead. Functions which execute in much less than a microsecond are not going to see a major improvement, as the wrapper code which transitions from the Python interpreter to Numba takes longer than a pure Python function call.

Now, I decided to test `numba` with matrix multiplication operation rather than `cosine similarity` implemented with 3 NumPy functions. Results are satisfying.

![](https://cdn1.imggmi.com/uploads/2018/5/26/1dcef8227c6f38e30b53109ead00b4bb-full.png)  

Matrix power operation:  
![](https://cdn1.imggmi.com/uploads/2018/5/26/ddeb10257b89c0e3c7013a95016dcd2d-full.png)

# Cython benchmarks
> The Cython language is a superset of the Python language that additionally supports calling C functions and declaring C types on variables and class attributes. This allows the compiler to generate very efficient C code from Cython code. The C code is generated once and then compiles with all major C/C++ compilers in CPython 2.6, 2.7 (2.4+ with Cython 0.20.x) as well as 3.3 and all later versions.

Before starting these benchmarks, I planned comparing pure python's performance with Cython's performance on calculating cosine similarity. But I found [another blog post](https://jakevdp.github.io/blog/2013/06/15/numba-vs-cython-take-2/#Comparing-the-Results) comparing `numba`, `cython`, `scikitlearn`, `SciPy` on pairwise distance functions. The blog post's conclusion is:
> Out of all the above pairwise distance methods, unadorned Numba is the clear winner, with highly-optimized Cython coming in a close second. Both beat out the other options by a large amount.

![Reference 1](https://cdn1.imggmi.com/uploads/2018/5/25/9b81c731cce1ec3ea03766ce4f159759-full.png)


> Note that this is log-scaled, so the vertical space between two grid lines indicates a factor of 10 difference in computation time!

Considering that `Cython` is a static compiler, we need
  * to write our code with `cython` syntax
  * build it
  * import from another python module

Before starting developing some complex operation in `cython`, I recognized that these steps require much more work than `numba`, a dynamic compiler.

Cython enables us to write C like python code with pointers, explicit types, arrays, structs, enums ...

In `cymatrix.pyx`, I implemented matrix multiplication like that:
```
import numpy as np

cpdef double[:,:] multiply_matrices(double[:,:] m1, double[:,:] m2):
  cdef int M = m1.shape[0]
  cdef int N = m1.shape[1]
  cdef int P = m2.shape[0]
  cdef int Q = m2.shape[1]
  cdef int c, d, k
  cdef double[:,:] res = np.zeros((M, Q))
  cdef double sum = 0

  for c in range(M):
    for d in range(Q):
      for k in range(N):
        sum += m1[c][k] * m2[k][d]

      res[c][d] = sum
      sum = 0

  return res
```
`setup.py` contains following lines:
```
from distutils.core import setup
from Cython.Build import cythonize

setup(name="cymatrix", ext_modules=cythonize('cymatrix.pyx'),)
```
and we compile from the terminal with
```
python setup.py build_ext --inplace
```
Then, I imported `cymatrix` module from my python test script and compared its performance with the same function written in python and auto-jitted numba version. Here is the result:

![](https://cdn1.imggmi.com/uploads/2018/5/26/f8a3e81822cfe433a04c4cb973d0aaf2-full.png)

My conclusion is that, `numba` and `cython` performs nearly same since both of them compile the `python` source code into C/C++ code and probably runs the similar C/C++ code snippet. Numba automatically creates compiled source code with single decorator `jit`, while we're explicitly re-writing python functions with types in `Cython`, building and importing. It's crystal clear that using dynamic compiler `Numba` is much easier for developers (us) than rewriting the same code in `Cython` with additional stuff. Supported by the results (see Reference 1) that I shared from the blog post, Numba beats Cython with equal performance and its dynamic, auto-type-detecting features.

# General Conclusion
### Parallelization
  * Use multiple threads only for IO tasks
  * Using multiple processes increases the performance, but this increase isn't linear with number of processes
  * Using more processes than number of cores in the machine does not effect the performance. Generally the performance stays same after running more processes than number of cores.

### Numba and Cython
  * Numba is a dynamic compiler and does not require to re-implement existing functions. Zero effort to integrate it into existing architecture. Cython is a static compiler and requires to re-develop the same python code in `Cython` language with `.pyx` extension. `.pyx` files are compiled with `setup.py` and imported by python scripts. Same functions compiled with `Numba` and `Cython` performs nearly same for most of the functions, but `Numba` is optimized for `NumPy` operations.
  * `NumPy` can multiply huge matrices within a few milliseconds. `NumPy` functions compiled with `Numba` even does these multiplications up to 50 times faster. Using for loops to complete the same operation takes more than a minute with native python code. But same for loops compiled with `Numba` takes less than a second. **`NUMBA` ROCKS!**  

For example  
* Input: 500 x 500 Matrix
* Operation: Matrix Multiplication

1. NumPy's matrix multiplication function: **13 milliseconds**
2. NumPy's matrix multiplication compiled with Numba: **0.6 milliseconds**
3. Matrix multiplication with for loops in python: **106 seconds**
4. Matrix multiplication with for loops in python compiled with Numba: **0.21 seconds**
5. Matrix multiplication with for loops written in Cython: **0.27 seconds**

  * Numba increased `NumPy` performance 22 times and `Python for loops`'s performance 504 times! We should definitely use `Numba` for mathematical functions and nested for loops. `Numba` can't make better for `string` operations as I showed in one of the previous benchmark results.
