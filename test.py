import liniter.solvers as solvers
import liniter.utils as utils
import numpy as np
import scipy.sparse as sp


# A = np.array([[1, -1, 0], [-1, 3, -1], [0, -1, 4]])
A = utils.load_custom_mtx("test_data/spa1.mtx")
# A = sp.tril(A)
# x = np.array([1, 2, 3])
x = np.ones(1000)

b = A.dot(x)

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


# A = sp.csr_matrix(A)

# print(A.toarray())

# print(A)

# print(A.indices)
# print(A.indptr)

# print(A.data)

solver = solvers.GaussSeidelSolver(A, b)
sol = solver.solve()
print(utils.relative_error(x_exact=x, x_approx=sol))
