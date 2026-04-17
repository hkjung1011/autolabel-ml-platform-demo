from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from app.plugins.retinex.config import MSRCRConfig


def apply_msrcr(image: Image.Image, config: MSRCRConfig) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) + 1.0
    msr = np.zeros_like(rgb, dtype=np.float32)

    for sigma in config.sigma_list:
        blurred = np.asarray(image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=sigma)), dtype=np.float32) + 1.0
        msr += np.log10(rgb) - np.log10(blurred)
    msr /= max(1, len(config.sigma_list))

    summed = np.sum(rgb, axis=2, keepdims=True)
    color_restoration = config.beta * (np.log10(config.alpha * rgb) - np.log10(summed + 1.0))
    msrcr = config.gain * (msr * color_restoration + config.offset / 255.0)

    out = np.zeros_like(msrcr, dtype=np.uint8)
    for channel in range(3):
        plane = msrcr[:, :, channel]
        low = np.percentile(plane, config.low_clip_percentile)
        high = np.percentile(plane, config.high_clip_percentile)
        if high <= low:
            normalized = np.clip(plane, 0, 255)
        else:
            normalized = (plane - low) / (high - low) * 255.0
        out[:, :, channel] = np.clip(normalized, 0, 255).astype(np.uint8)
    return out
