import numpy as np
import scipy.sparse as sp


def is_sparse(A):
    """
    Checks if matrix is in Compressed Row Format

    Args:
        A (scipy.sparse matrix): matrix to check

    Returns:
        bool: True if matrix is in Compressed Row Format, False otherwise
    """

    return isinstance(A, sp.csr_matrix)


def is_square(A):
    """
    Checks if matrix is square

    Args:
        A (scipy.sparse matrix): matrix to check

    Returns:
        bool: True if matrix is square, False otherwise
    """
    return A.shape[0] == A.shape[1]


def is_vector_compatible(A, b):
    """
    Checks if matrix A and vector b are compatible by comparing matrix A's rows with vector b's length.

    Args:
        A (scipy.sparse matrix): matrix to check
        b (numpy array): vector to check compatibility with A

    Returns:
        bool: True if matrix A and vector b are compatible, False otherwise
    """
    return A.shape[0] == b.shape[0]


def is_symmetric(A, rtol=1e-5, atol=1e-8):
    """
    Checks if matrix A is symmetric by comparing A with its transpose. The comparison is calculated within a tollerance given by |a − b| ≤ atol + rtol * |b|, to avoid floating point errors.

    Args:
        A (scipy.sparse matrix): matrix to check for symmetry
        rtol (float): relative tollerance
        atol (float): absolute tollerance

    Returns:
        bool: True if matrix A is symmetric, False otherwise
    """
    if not is_sparse(A):
        raise ValueError("Matrix A needs to be sparse.")
    if not is_square(A):
        raise ValueError("Matrix A must be square to check for symmetry.")

    # Check if A and A transposed are equal
    diff = A - A.transpose()

    # != operator does not take into account a tollerance
    # so it could return False for very small values
    # for this reason, we convert to COO format and also
    # perform a check with tollerances
    diff_coo = diff.tocoo()
    # diff_coo.data is a numpy array
    # absolute(a - b) <= (atol + rtol * absolute(b))
    return np.allclose(diff_coo.data, 0, rtol=rtol, atol=atol)


def is_triangular(A, lower=True, rtol=1e-5, atol=1e-8):
    """
    Checks if matrix A is triangular by comparing A with its lower or upper triangular part. The comparison is calculated within a tollerance given by |a − b| ≤ atol + rtol * |b|, to avoid floating point errors.

    Args:
        A (scipy.sparse matrix): matrix to check for symmetry
        lower (bool): if True, check if A is lower triangular, otherwis check if upper triangular
        rtol (float): relative tollerance
        atol (float): absolute tollerance

    Returns:
        bool: True if matrix A is triangular (lower or upper), False otherwise
    """
    if not is_sparse(A):
        raise ValueError("Matrix A needs to be sparse.")
    if not is_square(A):
        raise ValueError("Matrix A must be square to check for triangularity check.")

    diff = A - sp.tril(A) if lower else sp.triu(A)
    diff_coo = diff.tocoo()
    return np.allclose(diff_coo.data, 0, rtol=rtol, atol=atol)


def is_diagonally_dominant(A):
    """
    Check if matrix A is diagonally dominant by checking if the absolute values of the diagonal elements are greater than the sum of the absolute values of the other elements in the same row.

    Args:
        A (scipy.sparse matrix): matrix to check for diagonal dominance

    Returns:
        bool: True if matrix A is diagonally dominant, False otherwise
    """

    if not is_sparse(A):
        raise ValueError("Matrix A needs to be sparse.")
    if not is_square(A):
        raise ValueError("Matrix A must be square for diagonal dominance check.")

    # Absolute values of A
    abs_A = abs(A)

    # Getting diagonal of A
    D = abs_A.diagonal()

    # Sum of elements on each row
    row_sums = abs_A.sum(axis=1).A1

    # Checking if diagonal element is bigger than the sum of the rest on the row
    S = row_sums - D
    return np.all(D > S)


def is_diagonal_non_zero(A, rtol=1e-5, atol=1e-8):
    """
    Check if the diagonal of matrix A is non-zero. The comparison is calculated within a tollerance given by |a − b| ≤ atol + rtol * |b|, to avoid floating point errors.

    Args:
        A (scipy.sparse matrix): matrix to check for non-zero diagonal elements
        rtol (float): relative tollerance
        atol (float): absolute tollerance

    Returns:
        bool: True if diagonal elements of A are non-zero, False otherwise.

    """
    
    if not is_sparse(A):
        raise ValueError("Matrix A needs to be sparse.")

    # Check if diagonal vector and zero vector are equal within tollerance
    return np.all(~np.isclose(A.diagonal(), 0, rtol=rtol, atol=atol))


def is_positive_definite(A):
    """
    Checks if matrix A is positive-definite, meaning that it is symmetric and all its eigenvalues are positive.

    Args:
        A (scipy.sparse matrix): matrix to check for positive-definiteness

    Returns:
        bool: True if matrix A is positive-definite, False otherwise
    """
    if not is_sparse(A):
        raise ValueError("Matrix A needs to be sparse.")

    # We check positive-definiteness by checking symmetry
    # and getting only the smallest eigenvalue and checking if it is positive,
    # meaning that all the other ones are also positive
    return (
        is_symmetric(A)
        and sp.linalg.eigsh(A, k=1, which="SA", return_eigenvectors=False) > 0
    )


def condition_number(A):
    """
    Computes the condition number of a sparse matrix A. A needs to be positive-definite.
    The condition number is estimated as the ratio of largest and smallest eigenvalues of A.
    Args:
        A (scipy.sparse matrix): matrix to compute condition number on

    Returns:
        float: condition number of matrix A
    """
    if not is_sparse(A):
        raise ValueError("Matrix A needs to be sparse.")
    if not is_positive_definite(A):
        raise ValueError(
            "Matrix A needs to be positive-definite to estimate condition number."
        )

    eig_max = sp.linalg.eigsh(A, k=1, which="LM", return_eigenvectors=False)[0]
    eig_min = sp.linalg.eigsh(A, k=1, which="SM", return_eigenvectors=False)[0]
    return eig_max / eig_min

def sparsity(A):
    """
    Computes the percentage of non-zero elements in the sparse matrix A by dividing number  of non-zero elements by the total number of elements in the matrix.
    Args:
        A (scipy.sparse matrix): sparse matrix to compute sparsity percentage on
    Returns:
        float: sparsity percentage of A
    """
    if not is_sparse(A):
        raise ValueError("Matrix A needs to be sparse.")

    return A.nnz / (A.shape[0] * A.shape[1]) * 100