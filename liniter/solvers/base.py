from abc import ABC, abstractmethod
import time
from ..utils import is_square, is_vector_compatible
from ..utils import error_message


class Solver(ABC):

    def __init__(self, A, b):
        self.A = A
        self.b = b

    def _check_matrix(self):
        # Check is matrix is square
        if not is_square(self.A):
            raise ValueError(error_message("Matrix A needs to be square."))
        if not is_vector_compatible(self.A, self.b):
            raise ValueError(
                error_message("Vector b does not match the size of the matrix A.")
            )

    @abstractmethod
    def _solve(self):
        pass

    def solve(self, print_time=True):
        self._check_matrix()
        time_start = time.time()
        sol = self._solve()
        time_end = time.time()
        if print_time:
            tot_time = time_end - time_start
            print(f"Elapsed time (s): {tot_time:.6f}")
        return sol


class IterativeSolver(Solver):

    def __init__(self, A, b, max_iter=20000, tol=1e-5):
        self.max_iter = max_iter
        self.tol = tol
        super().__init__(A, b)

    @abstractmethod
    def _solve(self):
        pass
