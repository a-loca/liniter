from abc import ABC, abstractmethod


class IterativeSolver(ABC):

    def __init__(self, A, b, max_iter=20000, tol=1e-5):
        self.A = A
        self.b = b
        self.max_iter = max_iter
        self.tol = tol

    def _check_matrix(self):
        # Check is matrix is square
        n_rows = self.A.shape[0]
        n_cols = self.A.shape[1]
        n = self.b.shape[0]

        if n_rows != n_cols:
            raise ValueError("Matrix A needs to be square.")
        if n_rows != n:
            raise ValueError("Vector b does not match the size of the matrix A.")

    @abstractmethod
    def _solve(self):
        pass

    def solve(self):
        self._check_matrix()
        return self._solve()
