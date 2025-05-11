import numpy as np
import scipy.sparse as sp

def is_diagonally_dominant(A):
    # Absolute values of A
    abs_A = abs(A)

    # Getting diagonal of A
    D = abs_A.diagonal()

    # Sum of elements on each row
    row_sums = abs_A.sum(axis=1).A1

    # Checking if diagonal element is bigger than the sum of the rest on the row
    S = row_sums - D
    return np.all(D > S)


def is_diagonal_non_zero(A):
    return np.all(A.diagonal() != 0)


def is_triangular(A, lower=True):
    if lower:
        return (A != sp.tril(A)).nnz == 0
    else:
        return (A != sp.triu(A)).nnz == 0


def is_symmetric(A, rtol=1e-5, atol=1e-8):
    # Check if A and A transposed are equal
    diff = A - A.transpose()
    if diff.nnz == 0:
        return True
    else:
        # != operator does not take into account a tollerance
        # so it could return False for very small values
        # for this reason, we convert to COO format and also
        # perform a check with tollerances
        diff_coo = diff.tocoo()
        # diff_coo.data is a numpy array
        return np.allclose(diff_coo.data, 0, rtol=rtol, atol=atol)


def is_positive_definite(A):
    # Matrix is positive-definite if symmetric and all its
    # eigenvalues are positive, by getting only the smallest eigenvalue
    # and checking if it is positive
    return (
        is_symmetric(A)
        and sp.linalg.eigsh(A, k=1, which="SA", return_eigenvectors=False) > 0
    )
