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
    pass
