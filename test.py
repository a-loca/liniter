import liniter.solvers as solvers
import liniter.utils as utils
import numpy as np
import scipy.sparse as sp
# import sys

# np.set_printoptions(threshold=sys.maxsize)

A = sp.csr_matrix(np.array([[0, -1, 0], [-1, 3, -1], [0, -1, 4]]))
# x = np.array([1, 1, 1])
# b = A.dot(x)

A = utils.load_custom_mtx("test_data/spa1.mtx")

print(A)
print(A.indices)
print(A.indptr)

# solver = solvers.GaussSeidelSolver(A, b)
# sol, _ = solver.solve()
# print(sol)
# print(x)
# print(utils.relative_error(x_exact=x, x_approx=sol))

print(utils.is_sparse(A))
