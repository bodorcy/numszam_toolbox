import numpy as np
import matplotlib.pyplot as plt

class Plotter:
    """
    Alap osztály, amely közös funkcionalitást biztosít a különböző típusú plottokhoz.
    """
    def __init__(self, f_name="Plot", data: list = None):
        self.data = [] if data is None else data.copy()
        self.f_name = f_name
        self.ax = None
        self.fig = None

    def plot_setup(self):
        """
        Közös setup minden plothoz.
        """
        if self.ax is None:
            self.fig = plt.figure(figsize=(6, 6))
            self.ax = self.fig.gca()
            self.ax.set_aspect('equal', adjustable='box')
            self.ax.grid(True)
            self.ax.axhline(0, color='gray', linewidth=0.5)
            self.ax.axvline(0, color='gray', linewidth=0.5)
            self.ax.set_title(self.f_name)

    def save(self, filename="figure.png"):
        """
        Elmenti a jelenlegi ábrát.
        :param filename: A fájl nev.
        """
        if self.fig is not None:
            self.fig.savefig(filename, bbox_inches='tight')
            print(f"Ábra elmentve: {filename}")
        else:
            print("Nincs menthető ábra.")

    def clear(self):
        """
        Tárolt adatok törlése.
        """
        self.data.clear()

    def show(self):
        """
        Ábra megjelenítése.
        """
        plt.show()


class VectorPlotter(Plotter):
    """
    Vektorok megjelenítése a 2D-s koordinátasíkon.
    """
    def __init__(self, f_name="Vektor plot", vectors: list = None):
        super().__init__(f_name, vectors)

    def plot_single(self, v: np.ndarray, hold=False):
        """
        Egyetlen vektor kirajzolása.
        :param v: 2D-s vektor
        :param hold: Ha True, ugyanara az ábrára rajzol.
        """
        self.plot_setup()

        if not hold:
            self.ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', alpha=1)
            plt.draw()

    def plot(self):
        """
        Minden tárolt vektor kirajzolása.
        """
        self.plot_setup()

        all_data = np.array(self.data)
        max_val = np.max(np.abs(all_data)) if len(all_data) > 0 else 1
        self.ax.set_xlim(-max_val * 1.2, max_val * 1.2)
        self.ax.set_ylim(-max_val * 1.2, max_val * 1.2)

        alpha = 0.7
        for i, v in enumerate(self.data):
            self.ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', alpha=alpha)
            self.ax.text(v[0] * 1, v[1] * 1, f"{i}", fontsize=10, alpha=alpha)
            plt.draw()
            alpha *= 0.9

        self.show()


class PolynomialPlotter(Plotter):
    """
    Polinomok megjelenítése.
    """
    def __init__(self, f_name="Polynomial plot", polynomials: list = None):
        super().__init__(f_name, polynomials)

    def plot_single(self, polynomial, hold=False):
        """
        Egyetlen polinom kirajzolása.
        :param polynomial: Ábrázolandó polinom
        :param hold: Ha True, ugyanarra az ábrára rajzol.
        """
        self.plot_setup()

        if not hold:
            x = np.linspace(-10, 10, 400)
            y = polynomial.evaluate(x)
            self.ax.plot(x, y, label="Polynomial", color='b')
            plt.draw()

    def plot(self):
        """
        Minden tárolt polinom kirajzolása.
        """
        self.plot_setup()

        x = np.linspace(-10, 10, 400)
        for poly in self.data:
            y = poly.evaluate(x)
            self.ax.plot(x, y, label="Polynomial", color='b')
            plt.draw()

        self.show()
