import liniter.solvers as solvers
from liniter.utils import relative_error, is_triangular, is_symmetric
import numpy as np

from liniter.solvers import LowerTriangularSolver, GaussSeidelSolver


A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 4]])
x = np.array([1, 2, 3])

b = A.dot(x)

print(A)
print(is_symmetric(A))

# A = np.array([[10, 2, 1], [1, 8, 1], [2, 1, 9]])
# x = np.array([1.0, 2.0, 3.0])
# b = A @ x


# solver = GaussSeidelSolver(A, b)
solver = solvers.GradientSolver(A, b)
sol = solver.solve()

print(sol)
print(relative_error(x, sol))
