from abc import ABC, abstractmethod
import time
from ..utils import is_square, is_vector_compatible, is_sparse
from ..utils import error_message


class Solver(ABC):

    def __init__(self, A, b, verbose=True):
        self.A = A
        self.b = b
        self.verbose = verbose

    def _check_matrix(self):
        if not is_sparse(self.A):
            raise TypeError(error_message("Matrix A is not sparse."))
        if not is_square(self.A):
            raise ValueError(error_message("Matrix A needs to be square."))
        if not is_vector_compatible(self.A, self.b):
            raise ValueError(
                error_message("Vector b does not match the size of the matrix A.")
            )

    @abstractmethod
    def _solve(self):
        pass

    def solve(self):
        # Check if matrix is in the correct format
        self._check_matrix()
        if self.verbose:
            # Starting time of the method
            time_start = time.time()
            # Solve linear system
            sol = self._solve()
            # Ending time
            time_end = time.time()
            # Find elapsed time
            tot_time = time_end - time_start
            print(f"Elapsed time (s): {tot_time:.6f}")
        else:
            sol = self._solve()

        return sol

class IterativeSolver(Solver):

    def __init__(self, A, b, verbose=True, max_iter=20000, tol=1e-4):
        self.max_iter = max_iter
        self.tol = tol
        super().__init__(A, b, verbose)

    @abstractmethod
    def _solve(self):
        pass
