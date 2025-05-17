from utils.plotter import PolynomialPlotter

"""
Polinom vektoros reprezentációja.
"""

import numpy as np

class Polynomial:
    def __init__(self, coeffs=None):
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
        if not (0 <= index <= len(self)):
            raise IndexError("Index out of range")
        return self.coeffs[index]

    def __setitem__(self, index, value):
        if not (0 <= index <= len(self)):
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
        prod = self.coeffs[0]

        for i in range(1, len(self)):
            prod = x0 * prod + self.coeffs[i]

        return prod



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
        result = np.zeros(n)
        poly_result = np.zeros(1)

        for i in range(n):
            xi, yi = self.points[i]
            numerator = np.array([1.0])  # polinom "1"
            denominator = 1.0

            for j in range(n):
                if i == j:
                    continue
                xj = self.points[j][0]
                # Multiply numerator by (x - xj)
                numerator = np.convolve(numerator, [1.0, -xj])  # konvolúció == polinomszorzás
                denominator *= (xi - xj)

            Li = numerator / denominator
            poly_result = np.pad(poly_result, (len(Li) - len(poly_result), 0))

            self.sub_polinomials.append(Polynomial(Li))

            poly_result += np.pad(Li * yi, (len(poly_result) - len(Li), 0))

        return poly_result

    def __str__(self):
        return (f"{[(float(p[0]), float(p[1])) for p in self.points]}\n"
                f"pontokhoz tartozó Lagrange polinom: {super().__str__()}")

    def interpolation_points(self):
        return self.points

lag_poly = LagrangePolynomial([(1, 2), (2, 3), (3, 5)])
print(lag_poly)              # Pretty-print polynomial
print(lag_poly.evaluate(2))  # Evaluate at x = 2 (should return ~3)
print(lag_poly.coeffs)       # Show polynomial coefficients
print(lag_poly.interpolation_points())  # Original points
print(lag_poly.sub_polinomials)


pplot = PolynomialPlotter(polynomials=lag_poly.sub_polinomials)
pplot.plot()