import numpy as np


def get_random_regular_matrix(n, min_val=-5, max_val=5):
    """
    Random n x n -es invertálható mátrix.
    """
    while True:
        A = np.random.randint(min_val, max_val+1, size=(n, n))
        print(np.linalg.det(A))
        if np.linalg.det(A) != 0:
            return A


def get_random_positive_def_matrix(n, min_val=-5, max_val=5):
    """
    Random n x n -es pozitív definit mátrix.
    """
    A = get_random_regular_matrix(n, min_val, max_val)
    return A @ A.T


def pretty_print_matrix(matrix, decimals=2):
    """
    Convert a NumPy matrix into a nicely formatted string.
    """
    rounded = np.round(matrix, decimals=decimals)
    rows = ["[" + "  ".join(f"{val:>6.2f}" for val in row) + "]"
            for row in rounded]
    return "\n".join(rows)
