"""
QR felbontás Gram-Schmidt eljárással.
"""

import numpy as np
from numpy import ndarray
from numpy.linalg import norm

def get_projection(u: ndarray, v: ndarray):
    """
    Gram-Schmidt eljárás-ban szereplő vektor projekció elvégzése.
    :param u: a vektor amiRE projektálunk
    :param v: a vekotr AMIT projektélunk
    :return:
        A v vektor u vektorra vett projekciója.
    """
    uv_skalarszorzat = np.dot(u, v)
    uu_skalarszorzat = np.dot(u, u)

    return (uv_skalarszorzat / uu_skalarszorzat) * u


def qr(A, verbose=False):
    """
    QR felbontás elvégzése, egy n x m -es mátrixon.
    :param A: n x m -es mátrix
    :param verbose: lépések kiiratása
    :return: Az A mátrix QR felbontása, (Q, R) alakban, ahol Q ortoginális, R felső trianguláris.
    """
    if verbose:
        print("------- QR felbontás -------")

    n, m = A.shape[0], A.shape[1]
    Q = np.zeros((n, m))  # Q n x m -es mátrix
    R = np.zeros((m, m))
    output = []

    if n != m or abs(np.linalg.det(A)) < 1e-5:
        print("Vigyázat, a mátrix (közel) szinguláris!")

    for j in range(m):
        v_oszlop = A[:, j]

        for i in range(j):
            R[i, j] = np.dot(Q[:, i], v_oszlop)  # R = Q' * A
            v_oszlop = v_oszlop - get_projection(Q[:, i], v_oszlop)

        # vektorokra kettes norma by default,
        # itt nagy numerikus hiba keletkezik, ha pl norm(v) == sqrt(2)
        R[j, j] = norm(v_oszlop)
        v_oszlop = v_oszlop / R[j, j]

        if verbose:
            output.append(f"Az A {j + 1}. oszlopának ortonormált vekotra:\n"
                  f"{v_oszlop}^T\n")

        Q[:, j] = v_oszlop

    output = "\n".join(output)

    return Q, R, output


def main():
    """
    asd
    """
    np.set_printoptions(
        precision=4,
        suppress=True,  #
        formatter={'float_kind': '{:.2f}'.format}
    )

    A = np.array([[0, 3], [2, 0]])

    Q, R = qr(A, verbose=True)

    print(Q @ R)


if __name__ == '__main__':
    main()
