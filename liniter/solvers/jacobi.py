from .base import IterativeSolver
import numpy as np
from ..utils import relative_residual
from ..utils import is_diagonally_dominant, is_diagonal_non_zero


class JacobiSolver(IterativeSolver):

    def _check_matrix(self):
        super()._check_matrix()
        # Check if diagonal has no null elements
        if not is_diagonal_non_zero(self.A):
            raise ValueError(
                "Matrix A has at least one zero on the diagonal. Jacobi failed."
            )
        # Check if matrix is diagonally dominant
        if not is_diagonally_dominant(self.A):
            print("WARNING: matrix is not diagonally dominant, Jacobi may fail.")

    def _solve(self):
        # Initializing solution vector
        x = np.zeros(self.A.shape[0])
        k = 0

        # Getting diagonal from A and creating a matrix
        A_diag = np.diag(self.A)
        D_inv = np.diag(1 / A_diag)

        # Upper + lower triangular matrices from A
        # This is - (L + U)
        LU = np.diag(A_diag) - self.A

        for k in range(self.max_iter):
            # If residual is lower than tollerance, then loop can end early
            if relative_residual(self.A, x, self.b) <= self.tol:
                print(f"Jacobi reached convergence after {k+1} iterations!")
                return x

            # Computing new solution
            # D x_k+1 = -(L+U) x_k +b
            x = D_inv @ (LU @ x + self.b)

        # Max number of iteration was reached without convergence
        print(
            f"Jacobi could not reach convergence after {self.max_iter} iterations, ending execution."
        )
        return x
