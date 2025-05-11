from .base import Solver
from ..utils import is_triangular, is_diagonal_non_zero
import numpy as np


class LowerTriangularSolver(Solver):
    def _check_matrix(self):
        super()._check_matrix()
        # Check if matrix is lower triangular
        if not is_triangular(self.A, lower=True):
            raise ValueError("Matrix A is not lower triangular")
        if not is_diagonal_non_zero(self.A):
            raise ValueError(
                "Matrix A has at least one zero on the diagonal. Lower Triangular solver failed."
            )

    def _solve(self):
        # Forward substitution method
        N = self.A.shape[0]
        x = np.zeros(N)
        diag = self.A.diagonal()

        # for i in range(1, N):
        #     x[i] = (self.b[i] - np.dot(self.A[i, :i], x[:i])) / diag[i]
        # return x

        for i in range(N):
            # Get start and end column indeces of i-th row
            start = self.A.indptr[i]
            end = self.A.indptr[i + 1]
            # Dot product between A and x
            sum = 0
            # Looping through non zero columns in the current row
            for j in range(start, end):
                # Getting indeces of non zero columns
                col = self.A.indices[j]
                # Getting values in the matrix for those columns
                a_ij = self.A.data[j]
                if col < i:
                    # Manual dot product
                    sum += a_ij * x[col]
            x[i] = (self.b[i] - sum) / diag[i]

        return x
