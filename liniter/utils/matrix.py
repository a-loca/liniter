import numpy as np


def is_diagonally_dominant(A):
    # Matrix with only diagonal coefficients
    D = np.diag(np.abs(A))
    # Sum of absolute values of all non-diagonal coefficients, for each row
    S = np.sum(np.abs(A), axis=1) - D
    # |a_ii| > sum |a_ij| for all i
    if np.all(D > S):
        return True
    else:
        return False

def is_diagonal_non_zero(A):
    return np.all(A.diagonal() != 0)
