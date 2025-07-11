from .base import IterativeSolver
from ..utils import is_symmetric, is_positive_definite, error_message, success_message
import numpy as np


class GradientSolver(IterativeSolver):
    def _check_matrix(self):
        super()._check_matrix()
        # Check if matrix is symmetric
        if not is_symmetric(self.A):
            raise ValueError(
                error_message("Matrix A is not symmetric, Gradient method failed.")
            )
        # Check if matrix is positive definite
        if not is_positive_definite(self.A):
            raise ValueError(
                error_message(
                    "Matrix A is not positive-definite, Gradient method failed."
                )
            )

    def _solve(self):
        x = np.zeros(self.A.shape[0])
        for k in range(0, self.max_iter):
            # Finding residual
            r = self.b - self.A @ x

            # Checking if error is below threshold
            if np.linalg.norm(r) / np.linalg.norm(self.b) <= self.tol:
                # Method has converged, return solution
                self.verbose and print(
                    success_message(
                        f"Gradient reached convergence after {k+1} iterations!"
                    )
                )
                return x

            # Computing learning rate
            alpha = (r.T @ r) / (r.T @ self.A @ r)

            # Computing step in the direction of -gradient
            # to minimize error function
            x = x + alpha * r

        # Method has not converged since for loop has ended
        self.verbose and print(
            f"Gradient could not reach convergence after {self.max_iter} iterations, ending execution."
        )
        return x
