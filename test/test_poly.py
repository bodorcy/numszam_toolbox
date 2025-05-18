import numpy as np
import pytest
from src.numszam_toolbox.poly import Polynomial, LagrangePolynomial, generate_base_points


def test_polynomial_init_and_str():
    p = Polynomial([1, 0, -2])
    assert str(p) == "1.00x^2 + -2.00"


def test_polynomial_length_and_indexing():
    p = Polynomial([3, 2, 1])
    assert len(p) == 3
    assert p[0] == 3
    assert p[2] == 1

    # context manageres assertion
    with pytest.raises(IndexError):
        _ = p[3]
    with pytest.raises(IndexError):
        _ = p[-1]


def test_polynomial_setitem():
    p = Polynomial([1, 2, 3])
    p[1] = 9
    assert p.coeffs[1] == 9


def test_polynomial_addition():
    p1 = Polynomial([1, 2])
    p2 = Polynomial([3])
    result = p1 + p2
    assert isinstance(result, Polynomial)
    assert np.allclose(result.coeffs, [1, 5])


def test_polynomial_multiplication():
    p1 = Polynomial([1, 1])  # (x + 1)
    p2 = Polynomial([1, -1])  # (x - 1)
    result = p1 * p2
    assert result == [1, 0, -1]  # x^2 - 1


def test_polynomial_evaluate():
    p = Polynomial([2, -6, 2])  # 2x^2 - 6x + 2
    assert np.isclose(p.evaluate(3), 2 * 9 - 6 * 3 + 2)


def test_lagrange_polynomial_basic():
    points = [(-1, 10), (0, 7), (1, 6)]
    lp = LagrangePolynomial(points)
    assert isinstance(lp, Polynomial)
    assert len(lp.coeffs) > 0

    for x, y in points:
        assert np.isclose(lp.evaluate(x), y, atol=1e-8)

    assert all(np.isclose(lp.coeffs, np.array([1, -2, 7]), atol=1e-8))


def test_lagrange_polynomial_subpolynomials():
    points = [(-1, 10), (0, 7), (1, 6)]
    lp = LagrangePolynomial(points)
    assert len(lp.sub_polinomials) == 3
    for poly in lp.sub_polinomials:
        assert isinstance(poly, Polynomial)

    assert all(np.isclose(lp.sub_polinomials[0], np.array([0.5, -0.5, 0]), atol=1e-8))
    assert all(np.isclose(lp.sub_polinomials[1], np.array([-1, 0, 1]),  atol=1e-8))
    assert all(np.isclose(lp.sub_polinomials[2], np.array([0.5, 0.5, 0]),  atol=1e-8))


def test_generate_base_points_length():
    base = generate_base_points(5)
    assert len(base) == 5
    assert all(isinstance(pt, tuple) and len(pt) == 2 for pt in base)
