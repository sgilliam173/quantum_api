import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from qiskit_aer import AerSimulator
from quantum_api_bench.app.main import app


# Override the backend resolution so API tests never call IBM
def mock_resolve_backend(backend_type: str):
    return AerSimulator(), "ideal"


@pytest.fixture
def client():
    with patch(
        "quantum_api_bench.app.api.routes._resolve_backend",
        side_effect=mock_resolve_backend
    ):
        yield TestClient(app)


class TestHealthEndpoint:

    def test_health_returns_ok(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestDeutschJozsaEndpoint:

    def test_valid_request_returns_200(self, client):
        response = client.post("/api/v1/deutsch-jozsa", json={
            "case": 1,
            "n_qubits": 1,
            "backend": "ideal",
            "shots": 256
        })
        assert response.status_code == 200

    def test_response_has_expected_fields(self, client):
        response = client.post("/api/v1/deutsch-jozsa", json={
            "case": 1, "shots": 256
        })
        data = response.json()
        for field in ["algorithm", "counts", "circuit_depth",
                      "runtime_ms", "correct_result"]:
            assert field in data, f"Missing field: {field}"

    def test_invalid_case_returns_500(self, client):
        response = client.post("/api/v1/deutsch-jozsa", json={"case": 99})
        assert response.status_code == 500

    def test_shots_reflected_in_response(self, client):
        response = client.post("/api/v1/deutsch-jozsa", json={
            "case": 1, "shots": 128
        })
        data = response.json()
        assert data["shots"] == 128


class TestGroverEndpoint:

    def test_valid_request_returns_200(self, client):
        response = client.post("/api/v1/grover", json={
            "target": "11",
            "backend": "ideal",
            "shots": 256
        })
        assert response.status_code == 200

    def test_response_correct_result_for_easy_target(self, client):
        response = client.post("/api/v1/grover", json={
            "target": "1", "shots": 512
        })
        assert response.json()["correct_result"] is True