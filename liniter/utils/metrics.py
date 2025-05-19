import numpy as np

def relative_error(x_exact, x_approx):
    """
    Computes the relative error between the exact solution and the approximate solution of a linear system. The relative error is defined as the quotient of the norm of the difference between the exact and approximate solutions and the norm of the exact solution.
    
    Args:
        x_exact (numpy array): exact solution vector
        x_approx (numpy array): approximate solution vector, computed by a solver
        
    Returns:
        float: relative error between the exact and approximate solutions.
    """
    return np.linalg.norm(x_approx - x_exact) / np.linalg.norm(x_exact)

def relative_residual(A, x_approx, b):
    """
    Computes the relative residual of a linear system Ax = b. The relative residual is defined as the quotient of the norm of the residual vector (Ax - b) and the norm of the vector b.
    
    Args:
        A (scipy.sparse matrix): matrix of coefficients of the linear system
        x_approx (numpy array): approximate solutin vector, computed by a solver
        b (numpy array): right-hand side vector of the linear system
    
    Returns:
        float: relative residual of the linear system. The smaller the value, the better the approximation of the solution by the solver.
    """
    return np.linalg.norm(A @ x_approx - b) / np.linalg.norm(b)
