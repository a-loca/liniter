from liniter.solvers import JacobiSolver
from liniter.utils import relative_error
import numpy as np


A = np.array([[4, 1, 1], [1, 3, 1], [1, 1, 5]])
x = np.array([1, 2, 3])

b = A.dot(x)

solver = JacobiSolver(A, b)
sol = solver.solve()

print(sol)
print(relative_error(x, sol))
