
"""
Polinom vektoros reprezentációja.
"""

import numpy as np


class Polynomial:
    def __init__(self, coeffs=None):
        """
        Polinom reprezentálására szolgáló osztály.
        :param coeffs: polinom együtthatói fokszám szerint csökkenő sorrendben (x^2 + 4) == [1, 0, 4]
        """
        self._coeffs = np.array([]) if coeffs is None else np.array(coeffs)

    @property
    def coeffs(self):
        return self._coeffs

    @coeffs.setter
    def coeffs(self, c):
        self._coeffs = c

    def __len__(self):
        return len(self.coeffs)

    def __str__(self):
        """Polinom pretty-print."""
        terms = []
        degree = len(self.coeffs) - 1
        for i, coeff in enumerate(self.coeffs):
            power = degree - i
            if coeff != 0:
                term = f"{coeff:.2f}" if power == 0 else f"{coeff:.2f}x^{power}"
                terms.append(term)
        return " + ".join(terms) if terms else "0"

    def __getitem__(self, index):
        if not 0 <= index <= len(self):
            raise IndexError("Index out of range")
        return self.coeffs[index]

    def __setitem__(self, index, value):
        if not 0 <= index <= len(self):
            raise IndexError("Index out of range")
        self.coeffs[index] = value

    def __mul__(self, other):
        """Polinomok szorzása (konvolúció)."""
        result = [0] * (len(self) + len(other) - 1)
        for i in range(len(self)):
            for j in range(len(other)):
                result[i + j] += self[i] * other[j]
        return result

    def __add__(self, other):
        """Két polinom összeadása."""

        degree_diff = len(self) - len(other.coeffs)
        if degree_diff > 0:
            other.coeffs = np.pad(other.coeffs, (degree_diff, 0))

        elif degree_diff < 0:
            self.coeffs = np.pad(self.coeffs, (-degree_diff, 0))

        return Polynomial(self.coeffs + other.coeffs)

    def evaluate(self, x0):
        """
        Polinom kiértékelése az x0 helyen, Horner-módszerrel.
        :param x0: polinom együtthatói
        """
        result = np.full_like(x0, self.coeffs[0], dtype=np.float64)  # tömb eval

        for coeff in self.coeffs[1:]:
            result = result * x0 + coeff

        return result


class LagrangePolynomial(Polynomial):
    def __init__(self, points):
        """
        Lagrange polinom konstruktor
        :param points: (x, y) tuples
        """
        self.points = np.array(points)
        self._sub_polinomials = []
        coeffs = self._compute_coefficients()
        super().__init__(coeffs)

    @property
    def sub_polinomials(self):
        return self._sub_polinomials

    def _compute_coefficients(self):
        """
        Lagrange polinom felépítése.
        """
        n = len(self.points)

        poly_result = np.zeros(1)

        for i in range(n):
            xi, yi = self.points[i]
            numerator = np.array([1.0])  # polinom "1"
            denominator = 1.0

            for j in range(n):
                if i == j:
                    continue
                xj = self.points[j][0]
                # multiply numerator by (x - xj)
                numerator = np.convolve(numerator, [1.0, -xj])  # konvolúció == polinomszorzás
                denominator *= (xi - xj)

            Li = numerator / denominator
            poly_result = np.pad(poly_result, (len(Li) - len(poly_result), 0))

            self._sub_polinomials.append(Polynomial(Li))

            poly_result += np.pad(Li * yi, (len(poly_result) - len(Li), 0))

        return poly_result

    def interpolation_points(self):
        return self.points


def generate_base_points(n: int) -> list:
    """
    Random 2D-s alappontok generálása. Bármely két pont x koordinátája különboző.
    :param n: generált alappntok száma
    :return: list of (int, int)
    """
    base_points = []
    domain = [x for x in range(-6, 6)]

    for i in range(n):
        base_points.append((domain.pop(np.random.randint(0, len(domain))),
                             np.random.randint(-6, 6)))
    return base_points
