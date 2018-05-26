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
