from .gradient import GradientSolver
from ..utils import success_message
import numpy as np


class ConjugateGradientSolver(GradientSolver):

    def _solve(self):
        x = np.zeros(self.A.shape[0])
        r = self.b - self.A @ x
        p = r
        for k in range(self.max_iter):
            ap = self.A @ p
            alpha = (p.T @ r) / (p.T @ ap)
            x = x + alpha * p
            r = r - alpha * ap

            # Checking if error is below threshold
            if np.linalg.norm(r) / np.linalg.norm(self.b) <= self.tol:
                # Method has converged, return solution
                print(
                    success_message(
                        f"Conjugate Gradient reached convergence after {k+1} iterations!"
                    )
                )
                return x

            ap_t = ap.T
            beta = (ap_t @ r) / (ap_t @ p)
            p = r - beta * p
        print(
            f"Conjugate Gradient could not reach convergence after {self.max_iter} iterations, ending execution."
        )
        return x
