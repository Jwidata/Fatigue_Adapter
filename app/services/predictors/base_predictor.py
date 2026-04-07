from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from app.services.feature_builder import FeatureSequence


class BasePredictor(ABC):
    name: str = "base"

    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def predict(self, sequence: FeatureSequence) -> Optional[Tuple[float, float, float]]:
        """Return (x_norm, y_norm, confidence) or None."""
        raise NotImplementedError

    @abstractmethod
    def status(self) -> str:
        raise NotImplementedError
