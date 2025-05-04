from abc import ABC, abstractmethod

class IterativeSolver(ABC):
    def __init__(self, A, b, max_iter=20000, tol=1e-5):
        self.A = A
        self.b = b
        self.max_iter = max_iter
        self.tol = tol
    
    @abstractmethod
    def solve(self):
        """
        Solve the linear system Ax = b, using an iterative method.

        Returns:
            x: the solution vector
        """
        pass