from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_seed_and_pipeline_flow() -> None:
    seed_response = client.post("/api/assets/demo-seed")
    assert seed_response.status_code == 200
    assets = seed_response.json()["assets"]
    assert assets

    asset_id = assets[0]["id"]
    run_response = client.post(f"/api/pipeline/run/{asset_id}")
    assert run_response.status_code == 200
    payload = run_response.json()
    assert payload["asset"]["selected_candidate_id"] is not None
    assert payload["evaluation"]["map50"] > 0
