import numpy as np
import matplotlib
from src.numszam_toolbox.utils import Plotter, VectorPlotter, PolynomialPlotter
from src.numszam_toolbox.poly import Polynomial

matplotlib.use('Agg')  # GUI nélküli display a teszteléshez


# tmp_path egy ideiglenes path a teszteléshez
def test_plotter_basic_operations(tmp_path):

    plotter = Plotter("TestPlot")
    assert plotter.f_name == "TestPlot"
    assert plotter.data == []

    plotter.data.append("test")
    assert len(plotter.data) == 1
    plotter.clear()
    assert plotter.data == []

    plotter.plot_setup()
    test_file = tmp_path / "test_figure.png"
    plotter.save(test_file)
    assert test_file.exists()


def test_vector_plotter_add_and_plot():
    vp = VectorPlotter("vectest")
    assert len(vp.data) == 0

    v = np.array([1.0, 2.0])
    vp.add(v)
    assert len(vp.data) == 1
    assert np.allclose(vp.data[0], v)

    vp.plot_setup()
    vp.plot()


# teszt, csak hogy nem dob hibát
def test_vector_plotter_single_plot():
    vp = VectorPlotter()
    vp.plot_single(np.array([2, 3]), hold=False)


def test_polynomial_plotter_add_and_plot():
    pp = PolynomialPlotter("Poly Test")
    dummy_poly = Polynomial([1])

    pp.add(dummy_poly)
    assert len(pp.data) == 1
    assert isinstance(pp.data[0], Polynomial)

    pp.plot()


# teszt, csak hogy nem dob hibát
def test_polynomial_plotter_plot_points():
    pp = PolynomialPlotter("polyplotter")
    points = [(1, 2), (3, 4), (-1, -2)]
    pp.plot_points(points)
