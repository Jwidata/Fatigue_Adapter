from __future__ import annotations

from collections import deque
from typing import Deque, Optional

from app.adapters.gaze.base import GazeAdapter
from app.models.schemas import GazePoint


class TobiiGazeAdapter(GazeAdapter):
    def __init__(self):
        self._buffer: Deque[GazePoint] = deque(maxlen=1)

    def ingest_sample(self, timestamp: float, x: float, y: float, source: str = "tobii") -> None:
        self._buffer.append(GazePoint(timestamp=timestamp, x=x, y=y, source=source))

    def next_point(self, image_width: int, image_height: int, timestamp: float) -> Optional[GazePoint]:
        if not self._buffer:
            return None
        point = self._buffer[-1]
        point.timestamp = timestamp
        return point
