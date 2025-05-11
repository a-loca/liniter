import numpy as np
from scipy.sparse import coo_matrix


def load_custom_mtx(path):
    with open(path) as f:
        # Reading first line to get dimensions of sparse matrix
        nrows, ncols, nnz = map(int, f.readline().split())

        # Load remaining data
        data = np.loadtxt(f)

    # Getting columns
    row = data[:, 0].astype(int) - 1
    col = data[:, 1].astype(int) - 1
    val = data[:, 2]

    # Building COO matrix
    A = coo_matrix((val, (row, col)), shape=(nrows, ncols))

    print("Matrix loaded successfully!")

    # Converting to Compressed Sparse Row format for better handling later on
    return A.tocsr()
