from __future__ import annotations

from app.adapters.gaze.base import GazeAdapter
from app.models.schemas import GazePoint


class TobiiGazeAdapter(GazeAdapter):
    """Placeholder for Tobii SDK integration."""

    def next_point(self, image_width: int, image_height: int, timestamp: float) -> GazePoint | None:
        # TODO: Connect Tobii SDK stream to unified gaze schema.
        return None
