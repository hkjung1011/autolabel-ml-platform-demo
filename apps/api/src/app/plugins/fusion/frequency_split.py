from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def split_low_high(image: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    blurred = Image.fromarray(image_uint8).filter(ImageFilter.GaussianBlur(radius=max(0.1, sigma)))
    low = np.asarray(blurred, dtype=np.float32) / 255.0
    high = image - low
    return low, high


def normalize_signed_image(image: np.ndarray) -> np.ndarray:
    centered = image - float(image.min())
    scale = float(centered.max()) or 1.0
    return centered / scale
