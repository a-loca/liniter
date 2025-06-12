import argparse
import liniter.solvers as solvers
import liniter.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

plt.style.use(
    "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle"
)


def visualize_matrix(A, filename):
    plt.figure(figsize=(12, 12))
    plt.spy(A, markersize=0.05)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Matrix visualization saved at {filename}")


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
    parser.add_argument(
        "--mute",
        action="store_true",
        default=False,
        help="Turn off verbose option for solvers: logs will not be printed.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Visualize the matrix A.",
    )

    # If user specifies a solver, then all is False
    # Otherwise, run all solvers
    args = parser.parse_args()

    if args.all:
        args.all = True
    else:
        # Altrimenti attiva all solo se nessun metodo specificato
        args.all = not (
            args.jacobi or args.gauss_seidel or args.gradient or args.conjugate_gradient
        )

    # Load the matrix A from the provided file
    A = utils.load_custom_mtx(args.A)
    if args.visualize:
        visualize_matrix(A, args.A.replace(".mtx", ".png"))

    # Create solution vector x
    x = np.ones(A.shape[0])

    # Create vector b
    b = A @ x

    solvers_list = []
    if args.all or args.jacobi:
        solvers_list.append(
            solvers.JacobiSolver(
                A, b, max_iter=args.max_iter, tol=args.tol, verbose=not args.mute
            )
        )
    if args.all or args.gauss_seidel:
        solvers_list.append(
            solvers.GaussSeidelSolver(
                A, b, max_iter=args.max_iter, tol=args.tol, verbose=not args.mute
            )
        )
    if args.all or args.gradient:
        solvers_list.append(
            solvers.GradientSolver(
                A, b, max_iter=args.max_iter, tol=args.tol, verbose=not args.mute
            )
        )
    if args.all or args.conjugate_gradient:
        solvers_list.append(
            solvers.ConjugateGradientSolver(
                A, b, max_iter=args.max_iter, tol=args.tol, verbose=not args.mute
            )
        )

    print(f"\nSettings and matrix properties:")
    print(f"\tMaximum iterations: {args.max_iter}")
    print(f"\tTollerance: {args.tol}")
    print(f"\tMatrix size: {A.shape[0]}x{A.shape[1]}")
    print(f"\tMatrix condition number: {utils.condition_number(A):.3f}")
    print(f"\tMatrix sparsity percentage: {utils.sparsity(A):.2f}%")
    print(
        f"\tDiagonal dominance percentage: {utils.get_diagonal_dominance(A) / A.shape[0]:.2f}%"
    )

    for solver in solvers_list:
        print("\n===================================================\n")
        print(f"Running \033[1m{solver.__class__.__name__}\033[0m...\n")
        sol = solver.solve()
        print(f"Relative error: {utils.relative_error(x_exact=x, x_approx=sol)}")


if __name__ == "__main__":
    main()
