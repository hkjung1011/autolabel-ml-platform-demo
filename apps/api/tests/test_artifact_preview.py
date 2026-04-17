from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def test_artifact_preview_serves_image(tmp_path: Path) -> None:
    image_path = tmp_path / "preview.png"
    Image.new("RGB", (24, 24), (120, 40, 40)).save(image_path)

    response = client.get("/api/research/v1/artifacts/preview", params={"path": str(image_path)})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content
