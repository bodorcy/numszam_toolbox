"""
Hatványmódszer megvalísitás, kirajzolással.
"""

import numpy as np
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
                f"  b_{iteration} =\n{np.round(b_k1, 4)}\n" \
                f"  ||b_{iteration}|| = {np.round(b_k1_norm, 4)}\n" \
                f"  b_{iteration} normált =\n{np.round(b_k1_unit, 4)}\n" \
                f"  Sajátérték becslés = {np.round(b_k1_norm / np.linalg.norm(b_k), 4)}\n\n"

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
