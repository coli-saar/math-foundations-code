import numpy as np
from numpy.linalg import *

# Methods for generating linear algebra problems, cf. Steele 1997.

max_val = 3

def mk_det_matrix(n, r):
    """Generates a random n x n matrix with determinant +/- r and integer coefficients.
    Special case: r = 1 => inverse has only integer coefficients
    Special case: r = 0 => generate linearly dependent vectors
    """
    
    L = np.tril(np.random.randint(-max_val, max_val, size=(n,n)))
    np.fill_diagonal(L, ones(n)) # set diagonal entries to values with product 1

    U = np.triu(np.random.randint(-max_val, max_val, size=(n,n)))
    np.fill_diagonal(U, ones(n)) # set diagonal entries to values with product 1

    m = np.random.randint(n)
    L[m,m] = r

    P = mk_random_permutation_matrix(n)

    ret = P @ L @ U
    return ret
        
def mk_random_permutation_matrix(n):
    "Generate a random n x n permutation matrix."
    ret = np.zeros((n,n), dtype='int64')
    perm = np.random.permutation(n)
    ret[np.arange(n), perm] = 1
    return ret

def im(A):
    "Convert A into an integer matrix"
    return np.round(A).astype('int64')

def ones(n):
    "Returns vector of +/- 1 values of length n that multiply to 1"
    x = (2 * np.random.randint(0,2,size=n)) - 1
    return (x / np.prod(x)).astype('int64')


def make_unique_sle(n):
    "Generates a random system of linear equations with a unique solution. Returns a tuple A, b."
    A = mk_det_matrix(n,1)
    solution = np.random.randint(-max_val, max_val, size=n)
    b = A @ solution
    return A,b

