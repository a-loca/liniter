import liniter.solvers as solvers
import liniter.utils as utils
import numpy as np
import scipy.sparse as sp


# A = np.array([[2, -1, 0], [-1, 3, -1], [0, -1, 4]])
# x = np.array([1, 2, 3])

# b = A.dot(x)

# print(A)
# print(is_symmetric(A))

# A = np.array([[10, 2, 1], [1, 8, 1], [2, 1, 9]])
# x = np.array([1.0, 2.0, 3.0])
# b = A @ x


# solver = GaussSeidelSolver(A, b)
# solver = solvers.GradientSolver(A, b)
# sol = solver.solve()

# print(sol)
# print(relative_error(x, sol))


A = utils.load_custom_mtx("test_data/spa1.mtx")

# print(np.all(A.diagonal() != 0))
# print(utils.is_triangular(sp.tril(A)))
# print((A != A.transpose()).nnz == 0)
# print(utils.is_symmetric(A))
# print(sp.linalg.eigs(A))

# print(np.abs(A))

# print(np.abs(A).diagonal())

print(utils.is_diagonally_dominant(A))
