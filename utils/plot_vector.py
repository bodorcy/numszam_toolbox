import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray


class VectorPlotter:
    """
    Vektorok megjelenítése a 2D-s koordinátasíkon.
    """
    def __init__(self, f_name="Vektor plot", vectors: list=None):
        self.vectors = [] if vectors is None else vectors.copy()
        self.f_name = f_name
        self.ax = None
        self.fig = None

    def __add__(self, vector: ndarray):
        """
        2D-s vektor hozzáadása a megjelenítendő vektorokhoz.
        :param vector: 2x1-es vagy 1x2-es ndarray
        """
        if vector.shape != (2,1) and vector.shape != (1,2):
            raise ValueError("A vektor nem 2D-s.")

        vector = vector.T if vector.shape == (1,2) else vector

        self.vectors.append(vector)

    def plot_single(self, v: np.ndarray, hold=False):
        """
        Egyetlen vektor kirajzolása.
        :param v: 2D-s vektor
        :param hold: Ha True, ugyanarra az ábrára rajzol.
        """
        if not hold or self.ax is None:
            self.fig = plt.figure(figsize=(6, 6))
            self.ax = self.fig.gca()
            self.ax.set_aspect('equal', adjustable='box')
            self.ax.grid(True)
            self.ax.axhline(0, color='gray', linewidth=0.5)
            self.ax.axvline(0, color='gray', linewidth=0.5)
            self.ax.set_title(self.f_name)

            norm = np.linalg.norm(v)
            self.ax.set_xlim(-norm * 1.2, norm * 1.2)
            self.ax.set_ylim(-norm * 1.2, norm * 1.2)

        self.ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy',
                       scale=1, color='r', alpha=1)

        plt.draw()

    def plot(self):
        """
        Minden tárolt vektor kirajzolása.
        """
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_title(self.f_name)

        all_data = np.array(self.vectors)
        max_val = np.max(np.abs(all_data)) if len(all_data) > 0 else 1
        ax.set_xlim(-max_val * 1.2, max_val * 1.2)
        ax.set_ylim(-max_val * 1.2, max_val * 1.2)

        alpha = 0.7
        for i, v in enumerate(self.vectors):
            ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy',
                      scale=1, color='r', alpha=alpha)
            ax.text(v[0] * 1, v[1] * 1, f"{i}", fontsize=10, alpha=alpha)
            plt.draw()
            alpha*=0.9

        plt.show()

    def clear(self):
        """
        Tárolt vektorok törlése.
        """
        self.vectors.clear()

    def save(self, filename="figure.png"):
        """
        Elmenti a jelenlegi ábrát.
        :param filename: A fájl neve.
        """
        if self.fig is not None:
            self.fig.savefig(filename, bbox_inches='tight')
            print(f"Ábra elmentve: {filename}")
        else:
            print("Nincs menthető ábra.")
