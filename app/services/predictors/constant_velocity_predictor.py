from __future__ import annotations

from typing import Optional, Tuple

from app.services.feature_builder import FeatureSequence
from app.services.predictors.base_predictor import BasePredictor


class ConstantVelocityPredictor(BasePredictor):
    name = "constant_velocity"

    def available(self) -> bool:
        return True

    def predict(self, sequence: FeatureSequence) -> Optional[Tuple[float, float, float]]:
        points = sequence.points
        if len(points) < 2:
            return None
        prev = points[-2]
        curr = points[-1]
        dt = max(curr.timestamp - prev.timestamp, 1.0)
        vx = (curr.x - prev.x) / dt
        vy = (curr.y - prev.y) / dt
        pred_x = curr.x + vx * dt
        pred_y = curr.y + vy * dt
        return pred_x / sequence.width, pred_y / sequence.height, 0.4

    def status(self) -> str:
        return "baseline"
