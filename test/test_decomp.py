import numpy as np
from decomp import lu, qr, chol


def test_lu_decomposition():
    A = np.array([[4.0, 3.0], [6.0, 3.0]])
    L, U, _ = lu(A)

    assert np.allclose(L @ U, A), "L @ U != A"
    assert np.allclose(np.tril(L), L), "L nem alsó trianguláris"
    assert np.allclose(np.triu(U), U), "U nem felső trianguláris"


def test_qr_decomposition():
    A = np.array([[1.0, 1.0], [1.0, -1.0]])
    Q, R, _ = qr(A)

    assert np.allclose(Q @ R, A, atol=1e-6), \
        "Q @ R != A"
    assert np.allclose(Q.T @ Q, np.identity(Q.shape[1]), atol=1e-6), \
        "Q nem ortogonális"
    assert np.allclose(np.triu(R), R), \
        "R nem felső trianguláris"


def test_cholesky_decomposition():
    A = np.array([[25.0, 15.0, -5.0],
                  [15.0, 18.0,  0.0],
                  [-5.0,  0.0, 11.0]])

    L = chol(A)

    assert np.allclose(L @ L.T, A, atol=1e-6), "L @ L.T != A"
    assert np.allclose(np.tril(L), L), "L nem alsó trianguláris"
