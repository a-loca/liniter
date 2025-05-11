import argparse
import liniter.solvers as solvers
import liniter.utils as utils
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Runs all iterative solvers on provided matrix. Vector b is calculated as A @ x, solution vector is initialized as a vector of ones."
    )
    parser.add_argument(
        "A",
        help="Path to the '.mtx' file containing the matrix to be solved",
    )
    parser.add_argument(
        "-tol",
        help="Tollerance for the iterative solvers. Default is 1e-4",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "-max_iter",
        help="Maximum number of iterations for the iterative solvers. Default is 20000",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all solvers: Jacobi, Gauss-Seidel, Gradient, Conjugate Gradient.",
    )
    parser.add_argument(
        "--jacobi",
        action="store_true",
        default=False,
        help="Run Jacobi solver.",
    )
    parser.add_argument(
        "--gauss-seidel",
        action="store_true",
        default=False,
        help="Run Gauss-Seidel solver.",
    )
    parser.add_argument(
        "--gradient",
        action="store_true",
        default=False,
        help="Run Gradient solver.",
    )
    parser.add_argument(
        "--conjugate-gradient",
        action="store_true",
        default=False,
        help="Run Conjugate Gradient solver.",
    )

    # If user specifies a solver, then all is False
    # Otherwise, run all solvers
    args = parser.parse_args()

    if not (
        args.jacobi or args.gauss_seidel or args.gradient or args.conjugate_gradient
    ):
        args.all = True
    else:
        args.all = False

    # Load the matrix A from the provided file
    A = utils.load_custom_mtx(args.A)

    # Create solution vector x
    x = np.ones(A.shape[0])

    # Create vector b
    b = A @ x

    solvers_list = []
    if args.all or args.jacobi:
        solvers_list.append(
            solvers.JacobiSolver(A, b, max_iter=args.max_iter, tol=args.tol)
        )
    if args.all or args.gauss_seidel:
        solvers_list.append(
            solvers.GaussSeidelSolver(A, b, max_iter=args.max_iter, tol=args.tol)
        )
    if args.all or args.gradient:
        solvers_list.append(
            solvers.GradientSolver(A, b, max_iter=args.max_iter, tol=args.tol)
        )
    if args.all or args.conjugate_gradient:
        solvers_list.append(
            solvers.ConjugateGradientSolver(A, b, max_iter=args.max_iter, tol=args.tol)
        )

    print(f"Starting solvers with tollerance={args.tol} and max_iter={args.max_iter}")

    for solver in solvers_list:
        print("===================================================")
        print(f"Running {solver.__class__.__name__}...")
        sol = solver.solve()
        print(f"Error: {utils.relative_error(x_exact=x, x_approx=sol)}")


if __name__ == "__main__":
    main()
