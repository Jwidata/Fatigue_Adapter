from __future__ import annotations

from typing import List, Optional, Tuple

from app.models.schemas import GazePoint
from app.services.predictors.base_predictor import BasePredictor


class ConstantVelocityPredictor(BasePredictor):
    name = "constant_velocity"

    def available(self) -> bool:
        return True

    def predict(self, sequence: List[GazePoint]) -> Optional[Tuple[float, float, float]]:
        if len(sequence) < 2:
            return None
        prev = sequence[-2]
        curr = sequence[-1]
        dt = max(curr.timestamp - prev.timestamp, 1.0)
        vx = (curr.x - prev.x) / dt
        vy = (curr.y - prev.y) / dt
        pred_x = curr.x + vx * dt
        pred_y = curr.y + vy * dt
        return pred_x, pred_y, 0.4

    def status(self) -> str:
        return "baseline"
