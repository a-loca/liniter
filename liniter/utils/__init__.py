# utils/__init__.py
from .metrics import relative_error, relative_residual
from .matrix import is_diagonally_dominant, is_diagonal_non_zero, is_triangular, is_symmetric, is_positive_definite, is_square, is_vector_compatible, is_sparse
from .io import load_custom_mtx
from .log import error_message, warning_message, success_message