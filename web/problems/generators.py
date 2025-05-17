import numpy as np
import decomp.cholesky as cholesky
from decomp.lu import lu
from utils.matrix_utils import get_random_regular_matrix, get_random_positive_def_matrix

def generate_lu():
    n = np.random.randint(2, 4)

    A = get_random_regular_matrix(n)

    L,U = lu(A)

    return (
        f"Végezdd el az LU felbontást a következő mátrixon:\n"
        f"{str(A)}.",
        f"L =\n{str(L)}\nU =\n{str(U)}"
    )

def generate_qr():
    n = np.random.randint(2,4)
    A = get_random_regular_matrix(n)
    Q, R = np.linalg.qr(A)
    return (
        f"Végezdd el a QR felbontást a következő mátrixon:\n"
        f"{str(A)}.",
        f"Q =\n{str(Q)}\nR =\n{str(R)}"
    )
def generate_cholesky():
    n = np.random.randint(2, 4)
    A = get_random_positive_def_matrix(n)
    L = cholesky.chol(A)
    return (
        f"Végezdd el a Cholesky felbontást a következő mátrixon:\n{str(A)}\n",
        f"L =\n{str(L)}"
    )

def generate_problem(topic):
    generators = {
        "lu": generate_lu,
        "qr": generate_qr,
        "cholesky": generate_cholesky,
        "power": lambda: ("Power method example", "Solution"),
        "lagrange": lambda: ("Construct Lagrange polynomial for (1,2),(2,3),(3,5)", "P(x) = 0.5x² - 0.5x + 2")
    }

    if topic not in generators:
        return "Invalid topic", ""

    return generators[topic]()
