import liniter.solvers as solvers
import liniter.utils as utils
import numpy as np
import scipy.sparse as sp


A = sp.csr_matrix(np.array([[0, -1, 0], [-1, 3, -1], [0, -1, 4]]))
# x = np.array([1, 1, 1])
# b = A.dot(x)

# solver = solvers.GaussSeidelSolver(A, b)
# sol, _ = solver.solve()
# print(sol)
# print(x)
# print(utils.relative_error(x_exact=x, x_approx=sol))

print(utils.is_sparse(A))
