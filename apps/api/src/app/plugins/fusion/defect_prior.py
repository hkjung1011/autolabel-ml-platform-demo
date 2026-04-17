from __future__ import annotations

import numpy as np


def edge_defect_prior(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    gy, gx = np.gradient(gray)
    gradient = np.sqrt((gx**2) + (gy**2))
    gradient = gradient + np.abs(laplacian(gray))
    scale = float(gradient.max()) or 1.0
    return gradient / scale


def to_gray(image: np.ndarray) -> np.ndarray:
    return (0.299 * image[:, :, 0]) + (0.587 * image[:, :, 1]) + (0.114 * image[:, :, 2])


def laplacian(gray: np.ndarray) -> np.ndarray:
    up = np.roll(gray, 1, axis=0)
    down = np.roll(gray, -1, axis=0)
    left = np.roll(gray, 1, axis=1)
    right = np.roll(gray, -1, axis=1)
    return (up + down + left + right) - (4.0 * gray)
