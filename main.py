from liniter.solvers import JacobiSolver
from liniter.utils import relative_error
import numpy as np
from liniter.utils import is_triangular

from liniter.solvers import LowerTriangularSolver, GaussSeidelSolver


# A = np.array([[3, 1, 1], [1, 4, 1], [1, 1, 5]])
# A = np.tril(A)
# x = np.array([1, 2, 3])

# b = A.dot(x)


A = np.array([[10, 2, 1], [1, 8, 1], [2, 1, 9]])
x = np.array([1.0, 2.0, 3.0])
b = A @ x


solver = GaussSeidelSolver(A, b)
sol = solver.solve()

print(sol)
print(relative_error(x, sol))
