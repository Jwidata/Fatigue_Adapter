from __future__ import annotations

import random
from typing import Optional

from app.adapters.gaze.base import GazeAdapter
from app.models.schemas import GazePoint
from app.utils.geometry_utils import clamp_point


class SyntheticGazeAdapter(GazeAdapter):
    def __init__(self, mode: str = "normal"):
        self.mode = mode
        self._last = None

    def set_mode(self, mode: str):
        self.mode = mode

    def _next_normal(self, width: int, height: int):
        cx, cy = width * 0.5, height * 0.5
        jitter = min(width, height) * 0.15
        x = random.gauss(cx, jitter)
        y = random.gauss(cy, jitter)
        return clamp_point(x, y, width, height)

    def _next_overload(self, width: int, height: int):
        x = random.uniform(0, width - 1)
        y = random.uniform(0, height - 1)
        return x, y

    def _next_fatigue(self, width: int, height: int):
        if self._last is None:
            x = width * 0.2
            y = height * 0.2
        else:
            x, y = self._last
            drift = min(width, height) * 0.02
            x += random.uniform(-drift, drift)
            y += random.uniform(-drift, drift)
        return clamp_point(x, y, width, height)

    def next_point(self, image_width: int, image_height: int, timestamp: float) -> Optional[GazePoint]:
        if self.mode == "overload" or self.mode == "overloaded":
            x, y = self._next_overload(image_width, image_height)
        elif self.mode == "fatigue" or self.mode == "drifting":
            x, y = self._next_fatigue(image_width, image_height)
        else:
            x, y = self._next_normal(image_width, image_height)
        self._last = (x, y)
        return GazePoint(timestamp=timestamp, x=x, y=y, source="synthetic", mode=self.mode)
