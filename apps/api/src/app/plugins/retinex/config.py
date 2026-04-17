from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MSRCRConfig:
    sigma_list: list[float] = field(default_factory=lambda: [15.0, 80.0, 250.0])
    gain: float = 128.0
    offset: float = 30.0
    alpha: float = 125.0
    beta: float = 46.0
    low_clip_percentile: float = 1.0
    high_clip_percentile: float = 99.0


DEFAULT_MSRCR_CONFIG = MSRCRConfig()
