"""
LU felbontás.
"""

import numpy as np
from numpy import ndarray

from utils.matrix_utils import pretty_print_matrix


class EliminaciosMatrix:
    """
    A Gauss-elimináció során,
    egy oszlopból kinullázásából adódó eliminációs mátrix.
    """
    def __init__(self, matrix: ndarray=None, size=-1):
        """
        :param matrix:
            Inicializálás mátrix-szal.
        :param size:
            Inicializálás size x size méretű egységmátrix-szal.
        """
        if matrix is not None and size != -1:
            raise ValueError("Nem adható meg egyszerre mátrix és méret.")
        if matrix is not None:
            self._matrix = matrix
        elif size != -1:
            self._matrix = np.eye(size, dtype=float)

    @property
    def matrix(self):
        return self._matrix

    def __mul__(self, other):
        """
        Eliminációs mátrixok szorzása ( == kompozíciója)

        :param other: Eliminációs mátrix, amivel jobbról szorzunk.
        :return: A két eliminációs mátrix kompozíciója.
        """
        prod = self.matrix.copy()

        for i in range(prod.shape[0]):
            for j in range(prod.shape[1]):
                if i > j:
                    prod[i, j] = self.matrix[i, j] + other.matrix[i, j]

        return EliminaciosMatrix(prod)

    def __getitem__(self, key):
        return self._matrix[key]

    def __setitem__(self, key, value):
        self._matrix[key] = value

    def __str__(self):
        return str(self._matrix)

    def invert_elim(self):
        """
        Eliminációs mátrix invertálás
        (főátló alatti elemek ellentettjeit vesszük).
        Az eredeti mátrixot is invertálja!
        :return:
        Az invertált elminációs mátrix.
        """
        self._matrix = (self._matrix @ (-1 * np.eye(self._matrix.shape[1]))
                       + (2 * np.eye(self._matrix.shape[1])))

        # A eliminációs mátrixot megszorozzuk -1 * I-al,
        # majd a főátlóhoz 2-t adunk, hogy a végén 1-esek maradjanok ott.

        return self


def lu(A: ndarray, verbose=False):
    """
    Négyzetes mátrix LU felbontása részleges pivotálással.

    :param A: Négyzetes, nxn-es mátrix.
    :type A: numpy.ndarray

    :param verbose: Ha True, akkor kiírja a lépéseket.
    :type verbose: bool

    :returns: Egy háromtagú tuple (L, U, output), ahol:
              - L (ndarray): Alsó, nxn-es háromszög mátrix, a főátlóban 1-ekkel.\n
              - U (ndarray): Felső háromszögmátrix.\n
              - output (ndarray): Ha verbose == True, akkor köztes infót is kapunk

    :rtype: (ndarray, ndarray, list)

    :raises ValueError: Ha A nem négyzetes, vagy a pivotálás nem végezhető el.
    """

    if not isinstance(A, ndarray):
        raise ValueError("Az A mátrix nem megfelelő típusú.")
    if A.shape[0] != A.shape[1]:
        raise ValueError("A mátrix nem négyzetes!")

    if verbose:
        print("------- LU felbontás -------")

    output = []
    n = A.shape[0]  # már biztos, hogy négyzetes
    elim_matrixok = []
    U = A.copy()
    #P = np.eye(n) # permutációs mátrix

    for j in range(n):  # j == oszlop
        pivot_sor = np.argmax(np.abs(U[j:, j])) + j

        """if np.abs(U[pivot_sor, j]) < 1e-6:
            raise ValueError("A felbontás nem végezhető el stabilan, a mátrix (közel) szinguláris!")

        if pivot_sor != j:  # egy másik sorban van megfelelő pivot elem
            U[[j, pivot_sor], :] = U[[pivot_sor, j], :]  # sorcsere hackkelés
            P[[j, pivot_sor], :] = P[[pivot_sor, j], :]
"""
        M = EliminaciosMatrix(matrix=None, size=n)
        pivot = U[j, j]

        # elimináció
        for i in range(j+1, U.shape[1]):
            M[i, j] = - U[i, j] / pivot

        elim_matrixok.insert(0, M)  # mátrixszorzásnak megfelelő sorrendben
        U = M.matrix @ U  # @ == mátrix szorzás

    Ls = [m.invert_elim() for m in elim_matrixok]
    Ls.reverse()
    L = EliminaciosMatrix(size=n)

    for l in Ls:
        L = L * l

    if verbose:
        output.append("Eliminációs mátrixok\n(Mn * Mn-1 * Mn-2 * ... * M1 * A):\n")
        n = len(elim_matrixok)
        for m in elim_matrixok:
            output.append(f"M{n}\n{pretty_print_matrix(m.matrix)}\n")
            n -= 1

    return L.matrix, U, output


def main():
    """asd"""
    np.set_printoptions(
        precision=4,
        suppress=True,  #
        formatter={'float_kind': '{:.2f}'.format}
    )

    # A = np.array([[1, 1, 1, 1], [2, 1, 1, 3], [3, 1, 3, 2], [1, 1, 5, 3]], dtype=float)

    A = np.array([[1, 1, 0], [2, 1, 0], [3, 1, 1]])

    L, U, outputstr = lu(A, verbose=True)

    # Check reconstruction
    reconstructed_A = A
    LU = L @ U
    print("Reconstructed A (P * A):\n", reconstructed_A)
    print("L * U:\n", LU)
    print("Difference between P A and L * U:\n", reconstructed_A - LU)


if __name__ == "__main__":
    main()
