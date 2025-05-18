"""
Cholesky felbontás.
"""

import numpy as np
from numpy import ndarray, transpose
from numpy.linalg import eigh


def chol(A: ndarray):
    """
    Cholesky felbontás elvégzése, szimmetrikus, pozitív definit mátrixon.

    :param A: szimmetrikus, pozitív definit mátrix
    :return: L alső trianguláris mátrix, ahol A = L * L^T
    """

    if not np.all(A == transpose(A)):
        raise ValueError("Az A mátrix nem szimmetrikus!")
    if not np.all(eigh(A)[0] > 0):  # eigh, mert szimmetrikus
        raise ValueError("Nem minden sajátérték > 0.")

    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i + 1):
            sum_k = np.dot(L[i, :j], L[j, :j])

            if i == j:
                L[i, j] = np.sqrt(A[i, i] - sum_k)
            else:
                L[i, j] = (A[i, j] - sum_k) / L[j, j]

    return L
