from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_includes_runtime_identity() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "pid" in payload
    assert "version" in payload
    assert "frozen" in payload
    assert "executable" in payload
    assert "build_label" in payload
