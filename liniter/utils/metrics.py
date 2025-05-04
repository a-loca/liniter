import numpy as np

def relative_error(x_exact, x_approx):
    return np.linalg.norm(x_approx - x_exact) / np.linalg.norm(x_exact)

def relative_residual(A, x_approx, b):
    return np.linalg.norm(A @ x_approx - b) / np.linalg.norm(b)
