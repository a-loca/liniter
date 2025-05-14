########################################################################
# Playground for testing and experimenting with Liniter
########################################################################

# ==============================================================
# IMPORTS
# ==============================================================

import numpy as np
import scipy.sparse as sp
import liniter.solvers as solvers
import liniter.utils as utils

# Remove comments below to get the full printed outputs for
# vectors and matrices
# import sys
# np.set_printoptions(threshold=sys.maxsize)

# ==============================================================
# MATRIX AND VECTOR PREPARATION
# Matrix A needs to be in sparse format, solution x and right hand side
# vector b are numpy dense arrays
# ==============================================================

# Load matrix from .mtx file
A = utils.load_custom_mtx("test_data/spa1.mtx")

# OR: Create a custom sparse matrix with numpy
# N = 100
# Comment the above 'A' assignment to use this instead
# A = sp.csr_matrix(np.random.rand(N, N))

# Right-hand side vector and solution vector
x = np.ones(A.shape[0])
b = A.dot(x)

# ==============================================================
# MATRIX CHECKS
# The library implements some basic checks on provided matrices:
# is_square, is_sparse, is_symmetric, is_triangular, is_positive_definite
# is_diagonal_non_zero, is_diagonally_dominant
# ==============================================================

print("Matrix is square:", utils.is_square(A))
print("Matrix has non-zero diagonal:", utils.is_diagonal_non_zero(A))
print("Matrix is diagonally dominant:", utils.is_diagonally_dominant(A))

# ==============================================================
# SOLVING THE LINEAR SYSTEM
# Available iterative solvers:
# - JacobiSolver
# - GaussSeidelSolver
# - GradientSolver
# - ConjugateGradientSolver
# ==============================================================

solver = solvers.JacobiSolver(A, b, max_iter=20000, tol=1e-4, verbose=True)
solution = solver.solve()

# ==============================================================
# OUTPUT
# The solver methods already output the number of iterations taken
# to solve the system. Additionally, one can decide to save time elapsed
# and compute the relative error with regards to the provided exact solution
# to the system
# ==============================================================

print(f"Solution: {solution}")
print(f"Relative error: {utils.relative_error(x_exact=x, x_approx=solution)}")
