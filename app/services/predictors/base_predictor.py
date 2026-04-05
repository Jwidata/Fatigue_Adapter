from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from app.models.schemas import GazePoint


class BasePredictor(ABC):
    name: str = "base"

    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def predict(self, sequence: List[GazePoint]) -> Optional[Tuple[float, float, float]]:
        """Return (x, y, confidence) or None."""
        raise NotImplementedError

    @abstractmethod
    def status(self) -> str:
        raise NotImplementedError
