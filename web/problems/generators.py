import numpy as np
import decomp.cholesky as cholesky
from decomp.lu import lu
from decomp.qr import qr
from iterative.gauss_seidel import gauss_seidel
from iterative.power import power_method
from utils.matrix_utils import get_random_regular_matrix, get_random_positive_def_matrix, pretty_print_matrix
from poly.polynomial import LagrangePolynomial, generate_base_points
from utils.plotter import PolynomialPlotter


def generate_lu():
    n = np.random.randint(2, 4)

    A = get_random_regular_matrix(n)

    L, U, output = lu(A, verbose=True)

    output_str = "\n".join(output)

    return (
        f"Végezdd el az LU felbontást a következő mátrixon:\n"
        f"{pretty_print_matrix(A)}\n.",
        f"L =\n{pretty_print_matrix(L)}\n\nU =\n{pretty_print_matrix(U)}"
        f"\n\n{output_str}"

    )
def generate_qr():
    n = np.random.randint(2,4)
    A = get_random_regular_matrix(n)
    Q, R, output = qr(A, True)

    return (
        f"Végezdd el a QR felbontást a következő mátrixon:\n"
        f"{pretty_print_matrix(A)}\n.",
        f"Q =\n{pretty_print_matrix(Q)}\nR =\n{pretty_print_matrix(R)}"
        f"\n\n{output}"
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
        f"Hajts végre az alábbi kezdővektoron (x0) és mátrixon a hatványmódszermódszerrel {iterations} iterációt!\n"
        f"Mi a spektrálsugár közelítése ennyi itaráció után?\nx0 =\n{pretty_print_matrix(v0, 0)}\n"
        f"A =\n{pretty_print_matrix(A)}\n",
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
def generate_gauss_seidel():
    n = np.random.randint(2, 4)
    A = get_random_regular_matrix(n)
    b = np.random.randint(-10, 10, size=(n, 1))
    x0 = np.zeros((n, 1))
    iterations = 3

    steps = gauss_seidel(A, b, x0, iterations)

    solution_text = ""
    for i, step in enumerate(steps, 1):
        solution_text += f"Iteráció {i}:\n{pretty_print_matrix(step)}\n\n"

    return (
        f"Közelítsd alábbi lineáris egyenletrendszer megoldását Gauss-Seidel iterációval ({iterations} lépés):\n"
        f"A =\n{pretty_print_matrix(A)}\nb =\n{pretty_print_matrix(b)}\nx0 =\n{pretty_print_matrix(x0)}",
        solution_text
    )


def generate_problem(topic):
    generators = {
        "lu": generate_lu,
        "qr": generate_qr,
        "cholesky": generate_cholesky,
        "power": generate_power,
        "lagrange": generate_lagrange,
        "gauss_seidel": generate_gauss_seidel
    }

    if topic not in generators:
        return "Invalid topic", ""

    return generators[topic]()
