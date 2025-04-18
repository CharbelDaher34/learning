import pytest
from fastapi.testclient import TestClient
from app.api.main import app, API_KEY

client = TestClient(app)


def test_infer_classification():
    response = client.post(
        "/infer",
        json={
            "text": "SpaceX successfully launches Falcon 9 rocket",
            "task": "classification",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "task" in data
    assert "result" in data
    assert "confidence" in data
    assert data["task"] == "classification"


def test_infer_generation():
    response = client.post(
        "/infer", json={"text": "SpaceX successfully", "task": "generation"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "task" in data
    assert "result" in data
    assert data["task"] == "generation"


def test_retrain_unauthorized():
    response = client.post("/retrain", json={"task": "classification"})
    assert response.status_code == 401


def test_retrain_authorized():
    response = client.post(
        "/retrain", json={"task": "classification"}, headers={"X-API-Key": API_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


def test_invalid_task():
    response = client.post("/infer", json={"text": "test", "task": "invalid_task"})
    assert response.status_code == 500
