from .base import IterativeSolver
import numpy as np
from liniter.utils import relative_residual


class JacobiSolver(IterativeSolver):

    def _check_matrix(self):
        super()._check_matrix()
        # Check if diagonal has no null elements
        if np.any(self.A.diagonal() == 0):
            raise ValueError(
                "Matrix A has at least one zero on the diagonal. Jacobi failed."
            )
        # TODO: check if A is diagonally dominant

    def _solve(self):
        # Initializing solution vector
        x = np.zeros(self.A.shape[0])
        k = 0

        # Getting diagonal from A and creating a matrix
        A_diag = np.diag(self.A)
        D_inv = np.diag(1 / A_diag)

        # Upper + lower triangular matrices from A
        LU = np.diag(A_diag) - self.A

        # Iterating until convergence
        while relative_residual(self.A, x, self.b) > self.tol:
            # Computing new solution
            x = D_inv @ (LU @ x + self.b)
            k += 1

            # Checking number of iterations
            if k > self.max_iter:
                print(
                    f"Jacobi could not reach convergence after {k} iterations, ending execution."
                )
                return x

        print(f"Jacobi reached convergence after {k} iterations!")
        return x
