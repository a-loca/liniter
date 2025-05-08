from base import IterativeSolver
from liniter.utils import is_symmetric, is_positive_definite


class GradientSolver(IterativeSolver):
    def _check_matrix(self):
        super()._check_matrix()
        # Check if matrix is symmetric
        if not is_symmetric(self.A):
            raise ValueError("Matrix A is not symmetric, Gradient method failed.")
        # Check if matrix is positive definite
        if not is_positive_definite(self.A):
            raise ValueError(
                "Matrix A is not positive-definite, Gradient method failed."
            )

    def _solve(self):
        return
