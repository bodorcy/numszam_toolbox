import numpy as np
from iterative import gauss_seidel
from iterative import power_method


def test_gauss_seidel_convergence():
    A = np.array([[4.0, 1.0, 2.0],
                  [3.0, 5.0, 1.0],
                  [1.0, 1.0, 3.0]])
    b = np.array([4.0, 7.0, 3.0])
    x0 = np.zeros_like(b)
    steps = gauss_seidel(A, b, x0, iterations=25)
    final_estimate = steps[-1]

    expected = np.linalg.solve(A, b)  # egzakt megoldás
    assert np.allclose(final_estimate, expected, atol=1e-4), \
        "A Gauss-Seidel iteráció nem a megfelelő vektorhoz konnvergál"


def test_power_method_dominant_eigenvalue_and_vector():
    A = np.array([[2.0, 1.0],
                  [1.0, 3.0]])

    # numpy-os függvényhez hasonlítás
    true_eigenvalues, true_eigenvectors = np.linalg.eig(A)

    # domináns sajátérték és sajátvektor index
    idx = np.argmax(np.abs(true_eigenvalues))

    expected_value = true_eigenvalues[idx]
    expected_vector = true_eigenvectors[:, idx]

    v0 = np.array([1.0, 0.0])
    approx_value, approx_vector, _ = power_method(
        A, v0=v0, max_iterations=100, tol=1e-6)

    approx_vector /= np.linalg.norm(approx_vector)
    expected_vector /= np.linalg.norm(expected_vector)

    assert np.isclose(approx_value, expected_value, atol=1e-3), \
        "Nem megfelelő sajátértékhez konvergál"
    assert np.allclose(np.abs(approx_vector), np.abs(expected_vector), atol=1e-2), \
        "Nem a megfelelő sajátvektorhoz konvergál"
