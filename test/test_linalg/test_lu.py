import unittest
import numpy as np
from decomp.lu import lu  # adjust path as needed

class TestLUDecomposition(unittest.TestCase):

    def test_lu_reconstruction(self):
        A = np.random.rand(4, 4)
        P, L, U = lu(A)
        PA = P @ A
        LU = P.T @ L @ U
        self.assertTrue(np.allclose(A, LU, atol=1e-8), "LU szorzat nem egyezik meg P*A-val")

    def test_l_and_u_shape(self):
        A = np.random.rand(5, 5)
        _, L, U = lu(A)
        self.assertEqual(L.shape, A.shape, "L dimenziója hibás")
        self.assertEqual(U.shape, A.shape, "U dimenziója hibás")

    def test_l_is_lower_triangular(self):
        _, L, _ = lu(np.random.rand(5, 5))
        self.assertTrue(np.allclose(L, np.tril(L)), "L nem alsó háromszög mátrix")

    def test_u_is_upper_triangular(self):
        _, _, U = lu(np.random.rand(5, 5))
        self.assertTrue(np.allclose(U, np.triu(U)), "U nem felső háromszög mátrix")

    def test_permutation_matrix(self):
        P, _, _ = lu(np.random.rand(5, 5))
        identity = np.eye(P.shape[0])
        PT = P.T
        self.assertTrue(np.allclose(P @ PT, identity), "P nem permutációs mátrix")

if __name__ == '__main__':
    unittest.main()
