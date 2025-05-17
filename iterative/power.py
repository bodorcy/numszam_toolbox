"""
Hatványmódszer megvalísitás, kirajzolással.
"""

import numpy as np
import matplotlib.pyplot as plt
#from utils.plot_vector import VectorPlotter
from utils.plotter import VectorPlotter


def power_method(A, v0=None, max_iterations=50, tol=1e-5, verbose=False, draw=False):
    """
    Hatványmódszer a legnagyobb magnitudójú sajátérték kiszámításához.

    :param A: numpy.ndarray, négyzertes mátrix
    :param v0: numpy.ndarray, kezdővektor
    :param max_iterations: maximum iterációk száma
    :param tol: tolerancia
    :param verbose: köztes lépések megjelenítése
    :param draw: a lépések kirajzolás
    :return: (sajátérték, sajátvektor)
    """
    output = ""
    n, m = A.shape
    if n != m:
        raise ValueError("A mátrixnak négyzetesnek kell lennie.")

    b_k = np.random.rand(n) if v0 is None else v0
    vectors = [b_k]

    for iteration in range(max_iterations):
        b_k1 = A @ b_k

        # normalizálás, kettes norma
        b_k1_norm = np.linalg.norm(b_k1)
        if b_k1_norm == 0:
            raise ValueError("Nullvektor keletkezett, a mátrix (közel) szinguláris.")

        # egységhosszú vektor
        b_k1_unit = b_k1 / b_k1_norm

        if np.linalg.norm(b_k1_unit - b_k) < tol:
            if verbose:
                print(f"Konvergált {iteration} iteráció után.")
            break

        if verbose:
            output += \
                f"Iteráció {iteration}:\n" \
                f"  b_{iteration}         = {np.round(b_k1, 4)}\n" \
                f"  ||b_{iteration}||     = {np.round(b_k1_norm, 4)}\n" \
                f"  b_{iteration} normált = {np.round(b_k1_unit, 4)}\n" \
                f"  Sajátérték becslés    = {np.round(b_k1_norm / np.linalg.norm(b_k), 4)}\n\n"
            """
            print(f"b_{iteration} = {np.round(b_k1, 4)} | b_{iteration}_norm = {np.round(b_k1_norm, 4)} |"
                  f"b_{iteration} normalizált = {np.round(b_k1_unit, 4)} |"
                  f"sajátérték becslés: {np.round(b_k1_norm / np.linalg.norm(b_k), 4)}")
            """

        b_k = b_k1_unit

        if draw:
            vectors.append(b_k)

    # Rayleigh hányados
    eigenvalue = (b_k.T @ A @ b_k) / (b_k.T @ b_k)
    eigenvector = b_k

    if draw:
        plotter = VectorPlotter("Hatványmódszer", vectors)
        plotter.plot()

    return eigenvalue, eigenvector, output


def main():
    """ asd """
    A = np.array([[2, 1],
                  [1, 3]], dtype=float)

    D, V = np.linalg.eig(A)

    v_rho = V[:, np.argmax(D)]

    pltr = VectorPlotter("eigen")

    v0 = np.array([-v_rho[1]+1e-3, v_rho[0]])

    pltr.plot_single(v0)
    pltr.plot_single(v_rho, True)
    plt.show()
    pltr.save()

    eigenvalue, eigenvector, _ = power_method(A, v0=v0, verbose=True, draw=True)

    print("Legnagyobb sajátérték:", eigenvalue)
    print("Hozzátartozó sajátvektor:", eigenvector)


if __name__ == "__main__":
    main()
