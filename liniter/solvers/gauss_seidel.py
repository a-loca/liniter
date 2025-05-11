from .base import IterativeSolver
from ..utils import (
    is_diagonally_dominant,
    relative_residual,
    is_diagonal_non_zero,
)
from .lower_triangular import LowerTriangularSolver
import numpy as np
from scipy.sparse import tril


class GaussSeidelSolver(IterativeSolver):
    def _check_matrix(self):
        super()._check_matrix()
        # Check if diagonal has no null elements
        if not is_diagonal_non_zero(self.A):
            raise ValueError(
                "Matrix A has at least one zero on the diagonal. Gauss-Seidel failed."
            )
        # Check if matrix is diagonally dominant
        if not is_diagonally_dominant(self.A):
            print("WARNING: matrix is not diagonally dominant, Gauss-Seidel may fail.")

    def _solve(self):

        # Initializing solution vector
        x = np.zeros(self.A.shape[0])
        k = 0

        # Getting lower triangular matrix and diagonal
        L = tril(self.A, format="csr")

        # Getting upper triangular matrix
        U = self.A - L

        for k in range(self.max_iter):
            # If residual is lower than tollerance, then loop can end early
            if relative_residual(self.A, x, self.b) <= self.tol:
                print(f"Gauss-Seidel reached convergence after {k+1} iterations!")
                return x

            # Computing new solution
            # L x_k+1 = b - U x_k
            solver = LowerTriangularSolver(L, self.b - U @ x)
            x = solver.solve()

        # Max number of iteration was reached without convergence
        print(
            f"Gauss-Seidel could not reach convergence after {self.max_iter} iterations, ending execution."
        )
        return x
