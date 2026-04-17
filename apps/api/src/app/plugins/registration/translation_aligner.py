from __future__ import annotations

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None


def load_grayscale_array(path: str) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def estimate_translation(reference: np.ndarray, moving: np.ndarray, max_shift_px: int) -> tuple[int, int, float]:
    if reference.shape != moving.shape:
        raise ValueError(f"Shape mismatch: reference={reference.shape}, moving={moving.shape}")

    if cv2 is not None:
        cv2_result = _estimate_translation_cv2(reference=reference, moving=moving, max_shift_px=max_shift_px)
        if cv2_result is not None:
            return cv2_result

    reference_features = _mix_features(reference)
    moving_features = _mix_features(moving)
    coarse_dx, coarse_dy = _phase_correlation_shift(reference_features, moving_features)
    coarse_dx = int(np.clip(coarse_dx, -max_shift_px, max_shift_px))
    coarse_dy = int(np.clip(coarse_dy, -max_shift_px, max_shift_px))

    best_dx = coarse_dx
    best_dy = coarse_dy
    best_score = -1.0
    for dy in range(max(-max_shift_px, coarse_dy - 2), min(max_shift_px, coarse_dy + 2) + 1):
        for dx in range(max(-max_shift_px, coarse_dx - 2), min(max_shift_px, coarse_dx + 2) + 1):
            ref_crop, mov_crop = overlapping_crop(reference_features, moving_features, dx, dy)
            if ref_crop.size == 0 or mov_crop.size == 0:
                continue
            score = similarity_score(ref_crop, mov_crop)
            if score > best_score:
                best_score = score
                best_dx = dx
                best_dy = dy

    return best_dx, best_dy, float(best_score)


def overlapping_crop(reference: np.ndarray, moving: np.ndarray, dx: int, dy: int) -> tuple[np.ndarray, np.ndarray]:
    ref_y0 = max(0, dy)
    ref_y1 = min(reference.shape[0], reference.shape[0] + dy)
    ref_x0 = max(0, dx)
    ref_x1 = min(reference.shape[1], reference.shape[1] + dx)

    mov_y0 = max(0, -dy)
    mov_y1 = min(moving.shape[0], moving.shape[0] - dy)
    mov_x0 = max(0, -dx)
    mov_x1 = min(moving.shape[1], moving.shape[1] - dx)

    return reference[ref_y0:ref_y1, ref_x0:ref_x1], moving[mov_y0:mov_y1, mov_x0:mov_x1]


def similarity_score(reference_crop: np.ndarray, moving_crop: np.ndarray) -> float:
    ref_centered = reference_crop - float(reference_crop.mean())
    mov_centered = moving_crop - float(moving_crop.mean())
    denominator = float(np.linalg.norm(ref_centered) * np.linalg.norm(mov_centered))
    if denominator <= 1e-9:
        mae = float(np.mean(np.abs(reference_crop - moving_crop)))
        return 1.0 - mae
    return float(np.sum(ref_centered * mov_centered) / denominator)


def _mix_features(image: np.ndarray) -> np.ndarray:
    gradients = _gradient_magnitude(image)
    if float(gradients.max()) > 0:
        gradients = gradients / float(gradients.max())
    centered = image - float(image.mean())
    return (0.45 * centered) + (0.55 * gradients)


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(image)
    return np.sqrt((gx**2) + (gy**2))


def _phase_correlation_shift(reference: np.ndarray, moving: np.ndarray) -> tuple[int, int]:
    ref_freq = np.fft.fft2(reference)
    mov_freq = np.fft.fft2(moving)
    cross_power = ref_freq * np.conj(mov_freq)
    magnitude = np.abs(cross_power)
    cross_power /= np.maximum(magnitude, 1e-9)
    correlation = np.fft.ifft2(cross_power)
    peak_y, peak_x = np.unravel_index(np.argmax(np.abs(correlation)), correlation.shape)

    if peak_y > (reference.shape[0] // 2):
        peak_y -= reference.shape[0]
    if peak_x > (reference.shape[1] // 2):
        peak_x -= reference.shape[1]

    return int(peak_x), int(peak_y)


def _estimate_translation_cv2(reference: np.ndarray, moving: np.ndarray, max_shift_px: int) -> tuple[int, int, float] | None:
    ref32 = np.asarray(reference, dtype=np.float32)
    mov32 = np.asarray(moving, dtype=np.float32)
    try:
        (shift_x, shift_y), _ = cv2.phaseCorrelate(ref32, mov32)  # type: ignore[union-attr]
    except Exception:
        return None

    shift_x = float(np.clip(shift_x, -max_shift_px, max_shift_px))
    shift_y = float(np.clip(shift_y, -max_shift_px, max_shift_px))
    warp = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,  # type: ignore[union-attr]
        50,
        1e-5,
    )
    try:
        _, warp = cv2.findTransformECC(  # type: ignore[union-attr]
            ref32,
            mov32,
            warp,
            cv2.MOTION_TRANSLATION,  # type: ignore[union-attr]
            criteria,
        )
        shift_x = float(np.clip(warp[0, 2], -max_shift_px, max_shift_px))
        shift_y = float(np.clip(warp[1, 2], -max_shift_px, max_shift_px))
    except Exception:
        pass

    dx = int(round(shift_x))
    dy = int(round(shift_y))
    ref_crop, mov_crop = overlapping_crop(reference, moving, dx, dy)
    if ref_crop.size == 0 or mov_crop.size == 0:
        return None
    return dx, dy, similarity_score(ref_crop, mov_crop)
