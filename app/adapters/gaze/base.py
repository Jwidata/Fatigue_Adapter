from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from app.models.schemas import GazePoint


class GazeAdapter(ABC):
    @abstractmethod
    def next_point(self, image_width: int, image_height: int, timestamp: float) -> Optional[GazePoint]:
        raise NotImplementedError
