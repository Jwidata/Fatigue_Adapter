from __future__ import annotations

from app.adapters.gaze.base import GazeAdapter
from app.models.schemas import GazePoint


class MouseGazeAdapter(GazeAdapter):
    def next_point(self, image_width: int, image_height: int, timestamp: float) -> GazePoint | None:
        return None
