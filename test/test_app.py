import os
import shutil
import pytest
from src.numszam_toolbox_szte.web.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"<html" in response.data


def test_get_problem_valid_topic(client):
    response = client.post("/get_problem", json={"topic": "gauss"})
    assert response.status_code == 200
    data = response.get_json()
    assert "problem" in data
    assert "solution" in data


def test_get_problem_lagrange(client):
    os.makedirs("static", exist_ok=True)

    try:
        response = client.post("/get_problem", json={"topic": "lagrange"})
        assert response.status_code == 200
        data = response.get_json()
        assert "problem" in data
        assert "solution" in data
        assert "plot" in data
    finally:
        if os.path.exists("static"):
            shutil.rmtree("static")


def test_get_problem_invalid_topic(client):
    response = client.post("/get_problem", json={"topic": "unknown_topic"})
    assert response.status_code in [200, 400, 500]
