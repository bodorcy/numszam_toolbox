import numpy as np
import decomp.cholesky as cholesky
from decomp.lu import lu
from iterative.power import power_method
from utils.matrix_utils import get_random_regular_matrix, get_random_positive_def_matrix, pretty_print_matrix
from poly.polynomial import LagrangePolynomial, generate_base_points
from utils.plotter import PolynomialPlotter


def generate_lu():
    n = np.random.randint(2, 4)

    A = get_random_regular_matrix(n)

    L,U = lu(A)

    return (
        f"Végezdd el az LU felbontást a következő mátrixon:\n"
        f"{pretty_print_matrix(A)}\n.",
        f"L =\n{pretty_print_matrix(L)}\nU =\n{pretty_print_matrix(U)}"
    )
def generate_qr():
    n = np.random.randint(2,4)
    A = get_random_regular_matrix(n)
    Q, R = np.linalg.qr(A)
    return (
        f"Végezdd el a QR felbontást a következő mátrixon:\n"
        f"{pretty_print_matrix(A)}\n.",
        f"Q =\n{pretty_print_matrix(Q)}\nR =\n{pretty_print_matrix(R)}"
    )
def generate_cholesky():
    n = np.random.randint(2, 4)
    A = get_random_positive_def_matrix(n)
    L = cholesky.chol(A)
    return (
        f"Végezdd el a Cholesky felbontást a következő mátrixon:\n{pretty_print_matrix(A)}\n",
        f"L =\n{pretty_print_matrix(L)}"
    )
def generate_power():
    n = np.random.randint(2,4)
    iterations = np.random.randint(2, 4)
    A = get_random_regular_matrix(n)
    v0 = np.random.randint(-3, 3, size=(n, 1))

    eigenvalue, eigenvector, out = power_method(A, v0=v0, max_iterations=iterations+1, verbose=True)

    return (
        f"Hajts végre az alábbi kezdővektoron (x0) és mátrixon a hatványmódszermódszerrel {iterations} iterációt!\n\
        Mi a spektrálsugár közelítése ennyi itaráció után?\nx0 =\n{pretty_print_matrix(v0, 0)}\n\
        A =\n{pretty_print_matrix(A)}\n",
        f"Spektrálsugár közelítés: {np.round(eigenvalue, 2)}\nIterációk: {out}"
    )
def generate_lagrange():
    n = np.random.randint(2, 5)
    path = "static/pn.jpg"

    bps = generate_base_points(n)
    pn = LagrangePolynomial(bps)
    plotter = PolynomialPlotter()
    plotter.add(pn)

    x1 = min([p[0] for p in bps])
    x2 = max([p[0] for p in bps])

    plotter.plot(x1, x2)
    plotter.plot_points(bps)
    plotter.save(path)

    Li = pn.sub_polinomials

    Li_str = [str(p) for p in Li]
    Li_str = "\n".join(f"L{i+1} = {str(p)}" for i, p in enumerate(Li_str))

    return (
        f"Illessz Lagrange polinomot az alábbi alappontokra:\n"
        f"{bps}",
        f"Lagrange polinom:\n{str(pn)}\n"
        f"Részpolinomok:\n{Li_str}",
        path
    )


def generate_problem(topic):
    generators = {
        "lu": generate_lu,
        "qr": generate_qr,
        "cholesky": generate_cholesky,
        "power": generate_power,
        "lagrange": generate_lagrange
    }

    if topic not in generators:
        return "Invalid topic", ""

    return generators[topic]()
