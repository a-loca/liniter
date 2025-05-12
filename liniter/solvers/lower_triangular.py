from .base import Solver
from ..utils import is_triangular, is_diagonal_non_zero, error_message
import numpy as np


class LowerTriangularSolver(Solver):
    def _check_matrix(self):
        super()._check_matrix()
        if not is_triangular(self.A, lower=True):
            raise ValueError(error_message("Matrix A is not lower triangular"))
        if not is_diagonal_non_zero(self.A):
            raise ValueError(
                error_message(
                    "Matrix A has at least one zero on the diagonal. Lower Triangular solver failed."
                )
            )

    def _solve(self):
        # Forward substitution method
        N = self.A.shape[0]
        x = np.empty(N)
        diag = self.A.diagonal()

        data = self.A.data
        indices = self.A.indices
        ptr = self.A.indptr

        # Formula:
        # x[i] = (b[i] - np.dot(A[i, :i], x[:i])) / diag[i]

        for i in range(N):
            # Get start and end column indeces of i-th row
            start = ptr[i]
            end = ptr[i + 1]
            dot_sum = 0
            # Looping through non zero columns in the current row
            for j in range(start, end):
                # Getting indeces of non zero columns
                col = indices[j]
                # Getting values in the matrix for those columns
                a_ij = data[j]
                if col < i:
                    # Manual dot product
                    dot_sum += a_ij * x[col]
            x[i] = (self.b[i] - dot_sum) / diag[i]

        return x
