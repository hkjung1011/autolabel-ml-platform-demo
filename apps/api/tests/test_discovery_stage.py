from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def _write_image(path: Path, brightness: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (brightness, brightness, brightness)).save(path)


def test_discovery_and_stage_candidate(tmp_path: Path) -> None:
    scan_root = tmp_path / "external"
    dataset_root = scan_root / "ship_defect_lux_dataset"
    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"
    labels_train.mkdir(parents=True, exist_ok=True)

    _write_image(images_train / "hull_lux160_0001-1.png", 170)
    _write_image(images_train / "hull_lux80_0001-1.png", 90)
    _write_image(images_train / "hull_lux40_0001-1.png", 40)
    _write_image(images_train / "hull_lux160_0002-1.png", 172)
    _write_image(images_train / "hull_lux80_0002-1.png", 92)
    _write_image(images_train / "hull_lux40_0002-1.png", 42)
    (labels_train / "hull_lux160_0001-1.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    (labels_train / "hull_lux160_0002-1.txt").write_text("1 0.4 0.4 0.2 0.2\n", encoding="utf-8")

    discovery = client.post(
        "/api/research/v1/discovery/candidates",
        json={"scan_root": str(scan_root), "limit": 5, "min_images": 3},
    )
    assert discovery.status_code == 200
    payload = discovery.json()
    assert payload["candidates"]
    assert payload["candidates"][0]["dataset_root"] == str(dataset_root)

    workspace_root = tmp_path / "research_workspace"
    stage = client.post(
        "/api/research/v1/discovery/stage",
        json={
            "source_dataset_root": str(dataset_root),
            "workspace_root": str(workspace_root),
            "staged_name": "ship_candidate",
            "max_groups": 2,
            "prefer_labeled": True,
            "bootstrap_after_stage": True,
        },
    )
    assert stage.status_code == 200
    stage_payload = stage.json()
    assert stage_payload["selected_group_count"] == 2
    assert Path(stage_payload["staged_dataset_root"]).exists()
    assert Path(stage_payload["staged_workspace_root"]).exists()
    assert (Path(stage_payload["staged_workspace_root"]) / "manifests" / "summary.json").exists()
