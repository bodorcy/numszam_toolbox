import os
import shutil

from src.numszam_toolbox.web import generate_problem


def test_generate_lu():
    problem, solution = generate_problem("lu")
    assert isinstance(problem, str)
    assert isinstance(solution, str)
    assert "LU" in solution or "L =" in solution


def test_generate_qr():
    problem, solution = generate_problem("qr")
    assert isinstance(problem, str)
    assert isinstance(solution, str)
    assert "Q =" in solution and "R =" in solution


def test_generate_cholesky():
    problem, solution = generate_problem("cholesky")
    assert isinstance(problem, str)
    assert isinstance(solution, str)
    assert "L =" in solution


def test_generate_power():
    problem, solution = generate_problem("power")
    assert isinstance(problem, str)
    assert isinstance(solution, str)
    assert "Spektrálsugár" in solution


def test_generate_lagrange():
    os.makedirs("static", exist_ok=True)

    try:
        problem, solution, plot_path = generate_problem("lagrange")

        assert isinstance(problem, str)
        assert isinstance(solution, str)
        assert isinstance(plot_path, str)
        assert plot_path.endswith(".jpg")
        assert os.path.exists(plot_path)
    finally:
        if os.path.exists("static"):
            shutil.rmtree("static")


def test_generate_gauss_seidel():
    problem, solution = generate_problem("gauss_seidel")
    assert isinstance(problem, str)
    assert isinstance(solution, str)
    assert "Iteráció" in solution


def test_generate_invalid_topic():
    problem, solution = generate_problem("invalid_topic")
    assert problem == "Invalid topic"
    assert solution == ""
