from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pydicom


def load_dicom(path: str):
    return pydicom.dcmread(path)


def get_window(ds) -> Optional[Tuple[float, float]]:
    center = getattr(ds, "WindowCenter", None)
    width = getattr(ds, "WindowWidth", None)
    if center is None or width is None:
        return None
    if isinstance(center, (list, tuple)):
        center = center[0]
    if isinstance(width, (list, tuple)):
        width = width[0]
    return float(center), float(width)


def apply_window(pixels: np.ndarray, window: Optional[Tuple[float, float]]):
    if window is None:
        low, high = np.percentile(pixels, (5, 95))
    else:
        center, width = window
        low = center - width / 2
        high = center + width / 2
    clipped = np.clip(pixels, low, high)
    if high - low <= 1e-6:
        return np.zeros_like(clipped, dtype=np.uint8), (low, high)
    normalized = (clipped - low) / (high - low)
    return (normalized * 255).astype(np.uint8), (low, high)
