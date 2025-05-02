"""
LU felbontás.
"""

import numpy as np
from numpy import ndarray


class EliminaciosMatrix:
    '''
    A Gauss-elimináció során, egy oszlopból kinullázásából adódó eliminációs mátrix.
    '''
    def __init__(self, matrix: ndarray=None, size=-1):
        '''
        :param matrix:
            Inicializálás mátrix-szal.
        :param size:
            Inicializálás size x size méretű egységmátrix-szal.
        '''
        if matrix is not None and size != -1:
            raise ValueError("Nem adható meg egyszerre mátrix és méret.")
        if matrix is not None:
            self.matrix = matrix
        elif size != -1:
            self.matrix = np.eye(size, dtype=float)

    def __mul__(self, other):
        '''
        Eliminációs mátrixok szorzása ( == kompozíciója)
        :param other:
            Eliminációs mátrix, amivel jobbról szorzunk.
        :return:
            A két eliminációs mátrix kompozíciója.
        '''

        prod = self.matrix

        for i in range(prod.shape[0]):
            for j in range(prod.shape[1]):
                if i > j:  # csak a főátló alatti elemek
                    prod[i, j] = self.matrix[i, j] + other[i, j]
                    # darabonkénti szorzásnál működik
                    # mivel pontosan az egyik mátrix eleme nem nulla

        return EliminaciosMatrix(prod)

    def __getitem__(self, key):
        return self.matrix[key]

    def __setitem__(self, key, value):
        self.matrix[key] = value

    def __str__(self):
        return str(self.matrix)

    def invert_elim(self):
        '''
        Eliminációs mátrix invertálás
        (főátló alatti elemek ellentettjeit vesszük). Az eredeti mátrixot invertálja!
        :return:
        Az invertált elminációs mátrix.
        '''
        self.matrix = (self.matrix @ (-1 * np.eye(self.matrix.shape[1]))
                       + (2 * np.eye(self.matrix.shape[1])))

        # A eliminációs mátrixot megszorozzuk -1 * I-al,
        # majd a főátlóhoz 2-t adunk, hogy a végén 1-esek maradjanok ott.

        return self


def lu(A: ndarray, verbose=False):
    '''
    Négyzetes mátrix LU felbontása.

    Parameters
    ----------
    A : ndarray
        Négyzetes, nxn-es mátrix.

    Returns
    -------
    L : ndarray
        Alsó, nxn-es trianguláris mátrix, a főtátlóiban egyesekkel.

    U : ndarray
        Felső, nxn-es trianguláris mátrix.

    Raises
    ------
    ValueError
        Ha A nem négyzetes, vagy a pivotálás nem végezhető el.
    '''
    if not isinstance(A, ndarray):
        raise ValueError("Az A mátrix nem megfelelő típusú.")
    if A.shape[0] != A.shape[1]:
        raise ValueError("A mátrix nem négyzetes!")
    if np.any(A.diagonal() == 0):
        raise ValueError("0 pivot elem")  #TODO: partial pivoting

    elim_matrixok = []
    U = A.copy()

    for j in range(A.shape[0]):
        M = EliminaciosMatrix(matrix=None, size=A.shape[1])
        pivot = U[j, j]
        for i in range(j+1, U.shape[1]):
            M[i, j] = - U[i, j] / pivot

        elim_matrixok.insert(0, M)  # mátrixszorzásnak megfelelő sorrendben
        U = M.matrix @ U

    Ls = [m.invert_elim() for m in elim_matrixok]
    L = EliminaciosMatrix(size=A.shape[1])

    for l in Ls:
        L = L * l

    if verbose:
        print("Eliminációs mátrixok: (Mn * Mn-1 * Mn-2 * ... * M1 * A)")
        n = len(elim_matrixok)
        for m in elim_matrixok:
            print(f"M{n}\n{m.matrix}\n")
            n -= 1
        print(f"L:\n{L}\nU:\n{U}")

    return L, U


def main():
    '''asd'''
    np.set_printoptions(
        precision=4,  # 3 decimal places
        suppress=True,  # Don't use scientific notation # Wider lines  # Print large arrays fully
        formatter={'float_kind': '{:.2f}'.format}  # Custom float format
    )

    A = np.array([[1, 1, 1, 1], [2, 1, 1, 3], [3, 1, 3, 2], [1, 1, 1, 1]], dtype=float)

    lu(A, verbose=False)

if __name__ == "__main__":
    main()
