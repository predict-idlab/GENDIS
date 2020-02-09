import numpy as np
cimport numpy as np
import math
import cython
np.import_array()
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def _pdist(np.ndarray[DTYPE_t, ndim=2] A,
           list B,
           np.ndarray[DTYPE_t, ndim=2] result):

    cdef int i, j, k
    cdef int nA = len(A)
    cdef int nB = len(B)

    cdef float dist, min_dist

    for i in xrange(nA):
        for j in xrange(nB):
            if result[i, j] == 0:
                min_dist = np.inf
                for k in xrange(len(A[i]) - len(B[j]) + 1):
                    dist = np.sqrt(np.sum((A[i, k:k+len(B[j])] - B[j])**2))
                    if dist < min_dist:
                        min_dist = dist
                result[i, j] = min_dist

@cython.boundscheck(False)
@cython.wraparound(False)
def _pdist_location(np.ndarray[DTYPE_t, ndim=2] A,
                    list B,
                    np.ndarray[DTYPE_t, ndim=2] distances,
                    np.ndarray[DTYPE_t, ndim=2] locations):

    cdef int i, j, k
    cdef int nA = len(A)
    cdef int nB = len(B)

    cdef float dist, min_dist, loc

    for i in xrange(nA):
        for j in xrange(nB):
            if distances[i, j] == 0:
                min_dist = np.inf
                loc = 0
                for k in xrange(len(A[i]) - len(B[j]) + 1):
                    dist = np.sqrt(np.sum((A[i, k:k+len(B[j])] - B[j])**2))
                    if dist < min_dist:
                        min_dist = dist
                        loc = k/float(len(A[i]) - len(B[j]))
                distances[i, j] = min_dist
                locations[i, j] = loc
