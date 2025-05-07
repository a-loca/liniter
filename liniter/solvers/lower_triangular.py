from .base import Solver
from ..utils import is_triangular
import numpy as np


class LowerTriangularSolver(Solver):
    def _check_matrix(self):
        super()._check_matrix()
        # Check if matrix is lower triangular
        if not is_triangular(self.A, lower=True):
            raise ValueError("Matrix A is not lower triangular")

    def _solve(self):
        # Forward substitution method
        N = self.A.shape[0]
        x = np.zeros(N)
        x[0] = self.b[0] / self.A[0, 0]
        for i in range(1, N):
            x[i] = (self.b[i] - np.dot(self.A[i, :i], x[:i])) / self.A[i, i]
        return x
