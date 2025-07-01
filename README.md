# Liniter
**Liniter** (Linear systems, iterative solvers) is a Python library for solving sparse linear systems using iterative methods. It is developed with object-oriented principles to provide a modular, extensible, and efficient solution for numerical linear algebra problems.

## 📌 Features
- Supports several iterative solvers:
    - Jacobi
    - Gauss-Seidel
    - Gradient
    - Conjugate Gradient
- Modular OOP design: each solver is implemented as a class with a solve() method. Paramters of the system to be solved (`A`, `b`) are passed as arguments to the constructor.)
- Tools for matrix checks and error/residual computations.
- Support for custom .mtx matrix loading.
- Optional logging and visualization of matrix sparsity.
- Includes experimental launch script and testing framework.
- Includes testing matrices in `test_data/` directory.

## 🧱 Repository Structure
```
liniter/
│
├── solvers/                     # Contains all solver implementations
│   ├── base.py                  # Abstract base Solver and IterativeSolver classes
│   ├── lower_triangular.py      # LowerTriangularSolver class 
|   ├── jacobi.py                # JacobiSolver class
│   ├── gauss_seidel.py          # GaussSeidelSolver class
│   ├── gradient.py              # GradientSolver class
│   └── conjugate_gradient.py    # ConjugateGradientSolver class 
│
├── utils/
│   ├── io.py                    # Custom MatrixMarket parser
│   ├── log.py                   # Colored logging utilities
│   ├── matrix.py                # Matrix property checks
│   └── metrics.py               # Error and residual calculations
│
├── run_solvers.py              # CLI launcher for testing
├── playground.py               # Example script for solver usage
└── requirements.txt            # Mainly NumPy, SciPy, Matplotlib
```

## 🚀 Usage
1. Clone repository:
    ```bash
    git clone https://github.com/a-loca/liniter
    cd liniter
    ```
2. Install dependencies (in a virtual environment, optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
3. Run a solver:
    ```bash
    python run_solvers.py path/to/matrix.mtx --jacobi -tol 1e-6 -max_iter 5000
    ```


## ⚙️ CLI Options
Users can use the `run_solvers.py` script to run any of the implemented solvers, with various options:
| Argument               | Description                                 |
| ---------------------- | ------------------------------------------- |
| `A`                    | Path to the `.mtx` matrix file              |
| `--jacobi`             | Run Jacobi solver                           |
| `--gauss-seidel`       | Run Gauss-Seidel solver                     |
| `--gradient`           | Run Gradient solver                         |
| `--conjugate-gradient` | Run Conjugate Gradient solver               |
| `--all`                | Run all provided solvers on `A`             |
| `--tol TOL`            | Convergence tolerance (default: `1e-4`)     |
| `--max_iter N`         | Max number of iterations (default: `20000`) |
| `--mute`               | Disable logs and warnings                   |
| `--visualize`          | Plot the sparsity structure of the matrix   |
