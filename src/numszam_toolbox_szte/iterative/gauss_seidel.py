"""
Gauss Seisel iteráció.
"""

import numpy as np


def gauss_seidel(A, b, x0, iterations):
    """
    Gauss Seidel iteráció.
    :param A: A együtthatómátrix az Ax = b lineáris egyenletrendszerben
    :param b: b konstansvektor az Ax = b lineáris egyenletrendszerben
    :param x0: az iteráció kezdővektora
    :param iterations: az elvégzendő iterációk száma
    :return: az iterációk során kapott vektorok egy listában
    """
    n = A.shape[0]
    x = x0.copy()
    steps = []

    for k in range(iterations):

        x_new = x.copy()
        for i in range(n):

            # egyneltrendszer átrendezés
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]

        steps.append(x_new.copy())
        x = x_new

    return steps
