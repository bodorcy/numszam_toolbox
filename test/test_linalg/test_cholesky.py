import unittest
import numpy as np
from decomp.cholesky import chol  # Replace with the actual path to your chol function


class TestCholeskyDecomposition(unittest.TestCase):

    def test_known_decomposition(self):
        A = np.array([[4, 12, -16],
                      [12, 37, -43],
                      [-16, -43, 98]], dtype=float)
        L = chol(A)
        A_reconstructed = L @ L.T
        self.assertTrue(np.allclose(A, A_reconstructed, atol=1e-8),
                        "A != L @ L.T")

    def test_random_positive_definite_matrix(self):
        np.random.seed(42)
        B = np.random.rand(5, 5)
        A = B @ B.T  # guaranteed symmetric positive-definite
        L = chol(A)
        A_reconstructed = L @ L.T
        self.assertTrue(np.allclose(A, A_reconstructed, atol=1e-8),
                        "A véletlenszerű szimmetrikus poz. def. mátrix felbontása hibás")

    def test_non_symmetric_matrix(self):
        A = np.array([[1, 2], [0, 1]], dtype=float)  # not symmetric
        with self.assertRaises(ValueError, msg="Nem szimmetrikus mátrixot is elfogad"):
            chol(A)

    def test_non_positive_definite_matrix(self):
        A = np.array([[1, 2], [2, 1]], dtype=float)  # symmetric but not PD
        with self.assertRaises(ValueError, msg="Nem pozitív definit mátrixot is elfogad"):
            chol(A)

    def test_lower_triangular_result(self):
        A = np.array([[25, 15, -5],
                      [15, 18,  0],
                      [-5,  0, 11]], dtype=float)
        L = chol(A)
        self.assertTrue(np.allclose(L, np.tril(L)),
                        "L nem alsó háromszögmátrix")

    def test_identity_matrix(self):
        A = np.eye(4)
        L = chol(A)
        self.assertTrue(np.allclose(L, np.eye(4)),
                        "Az identitásmátrix felbontása hibás")

    def test_small_matrix(self):
        A = np.array([[2.0]])
        L = chol(A)
        self.assertTrue(np.allclose(L @ L.T, A),
                        "1x1-es mátrix felbontása hibás")


if __name__ == '__main__':
    unittest.main()
