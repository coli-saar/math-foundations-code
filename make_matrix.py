import numpy as np
from numpy.linalg import *
import math

# Methods for generating linear algebra problems, cf. Steele 1997.

#max_val = 4

def mk_det_matrix(n, r, rank=None, max_val=3):
    """Generates a random n x n matrix with determinant +/- r and integer coefficients.
    Special case: r = 1 => inverse has only integer coefficients
    Special case: r = 0 => generate linearly dependent vectors

    If the rank argument has a non-null value, the matrix is reduced
    to the given rank. r = 1 and rank < n has the advantage over r = 0
    that we still get only integer values when running the Gauss algorithm.
    """

    rank = rank or n
    
    L = np.tril(np.random.randint(-max_val, max_val, size=(n,n)))
    np.fill_diagonal(L, ones(n)) # set diagonal entries to values with product 1

    U = np.triu(np.random.randint(-max_val, max_val, size=(n,n)))
    np.fill_diagonal(U, ones(n)) # set diagonal entries to values with product 1

    m = np.random.randint(n)
    L[m,m] = r

    P = mk_random_permutation_matrix(n)

    down_ranker = np.zeros((n,n), dtype='int64')
    one_positions = np.random.choice(n, size=rank, replace=False)
    down_ranker[one_positions, one_positions] = 1
    # P = P @ down_ranker

    ret = P @ L @ down_ranker @  U
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


def make_unique_sle(n, max_val=3):
    "Generates a random system of linear equations with a unique solution. Returns a tuple A, b."
    A = mk_det_matrix(n,1)
    solution = np.random.randint(-max_val, max_val, size=n)
    b = A @ solution
    return A,b,solution

def make_unsolvable_sle(n, max_val=3):
    "Generates a random system of linear equations that has no solutions. Returns a tuple A, b."
    A = make_low_rank_matrix(n)
    
    b = np.random.randint(-max_val, max_val, size=n)
    while is_linear_combination(A, b):
        print("regenerate b")
        b = np.random.randint(-max_val, max_val, size=n)

    return A, b

def make_underconstrained_sle(n, max_val=3, max_solution_val=3):
    "Generates a random system of linear equations that has infinitely many solutions. Returns a tuple (A, b, solution), where solution is one of the solutions."
    A = make_low_rank_matrix(n)
    solution = np.random.randint(-max_solution_val, max_solution_val, size=n)
    b = A.dot(solution)
    return A, b, solution



def is_linear_combination(A, b):
    "Tests if b can be expressed as linear combination of the column vectors of A"
    aug = np.zeros((A.shape[0], A.shape[1]+1))
    aug[:, :-1] = A
    aug[:, -1] = b
    return np.linalg.matrix_rank(aug) == np.linalg.matrix_rank(A)


def make_low_rank_matrix(n):
    "Generate random interesting nxn matrix with rank n-1. Still guarantees (I think) that only integer values appear when applying the Gauss algorithm."
    A = mk_det_matrix(n, 1, rank=n-1)

    while(True):
        if is_interesting_matrix(A):
            return A
        else:
            A = mk_det_matrix(n, 1, rank=n-1)

def make_invertible_matrix(n):
    "Returns an invertible matrix with integer coefficients, with its inverse in integer coefficients."
    I = np.eye(n)

    while True:
        A = mk_det_matrix(n, 1)
        invA = inv(A).astype('int64')

        if np.array_equal(np.matmul(A, invA), I): # double-check for rounding errors
            return A, invA


def make_determinant_problem(n):
    while True:
        det = np.random.randint(-5, 6)
        A = mk_det_matrix(n, det)
        actual_determinant = int(round(np.linalg.det(A)))

        if abs(det) == abs(actual_determinant):
            return A, actual_determinant

def make_eigen_problem(n, max_abs_eigenvalue=2):
    choices = np.concatenate([np.arange(-max_abs_eigenvalue, 0), np.arange(1,1+max_abs_eigenvalue)])
    
    eigenvalues = np.random.choice(choices, n)
    D = np.diag(eigenvalues)      # nxn diagonal matrix of eigenvalues

    P, Pinv = make_invertible_matrix(n)    # nxn invertible matrix; columns are eigenvectors; change-of-basis from eigenbasis to standard basis

    A = P @ D @ Pinv
    return A, eigenvalues, P


def is_interesting_matrix(A):
    # check for zero rows
    n = A.shape[0]
    z = np.zeros(n, dtype=int)
    if any(np.array_equal(row, z) for row in A):
        return False

    # check for duplicate rows
    unique_rows = set([tuple(row) for row in A])
    if len(unique_rows) < n:
        return False

    return True

