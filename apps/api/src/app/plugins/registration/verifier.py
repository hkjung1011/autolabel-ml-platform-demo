from __future__ import annotations

from pathlib import Path


def read_yolo_boxes(label_path: Path) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _, x_center, y_center, width, height = parts[:5]
        boxes.append((float(x_center), float(y_center), float(width), float(height)))
    return boxes


def label_iou_drift(
    boxes: list[tuple[float, float, float, float]],
    image_width: int,
    image_height: int,
    dx_px: int,
    dy_px: int,
) -> float:
    if not boxes:
        return 0.0

    ious = []
    for x_center, y_center, width, height in boxes:
        original = to_xyxy(x_center, y_center, width, height, image_width, image_height)
        shifted = (
            original[0] + dx_px,
            original[1] + dy_px,
            original[2] + dx_px,
            original[3] + dy_px,
        )
        ious.append(iou(original, shifted))
    return 1.0 - (sum(ious) / len(ious))


def to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    box_width = width * image_width
    box_height = height * image_height
    x_mid = x_center * image_width
    y_mid = y_center * image_height
    x0 = x_mid - (box_width / 2.0)
    y0 = y_mid - (box_height / 2.0)
    x1 = x_mid + (box_width / 2.0)
    y1 = y_mid + (box_height / 2.0)
    return x0, y0, x1, y1


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])

    intersection_width = max(0.0, x1 - x0)
    intersection_height = max(0.0, y1 - y0)
    intersection_area = intersection_width * intersection_height

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection_area
    if union <= 1e-9:
        return 0.0
    return intersection_area / union
