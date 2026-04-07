from __future__ import annotations

import math
from typing import Optional, Tuple

from app.services.feature_builder import FeatureSequence
from app.services.predictors.base_predictor import BasePredictor


class HeuristicPredictor(BasePredictor):
    name = "heuristic"

    def available(self) -> bool:
        return True

    def predict(self, sequence: FeatureSequence) -> Optional[Tuple[float, float, float]]:
        points = sequence.points
        if len(points) < 3:
            return None
        velocities = []
        for prev, curr in zip(points, points[1:]):
            dt = max(curr.timestamp - prev.timestamp, 1.0)
            velocities.append(((curr.x - prev.x) / dt, (curr.y - prev.y) / dt))
        if not velocities:
            return None
        avg_vx = sum(v[0] for v in velocities) / len(velocities)
        avg_vy = sum(v[1] for v in velocities) / len(velocities)
        mean_speed = math.sqrt(avg_vx**2 + avg_vy**2)
        speed_var = sum(((math.sqrt(vx**2 + vy**2) - mean_speed) ** 2) for vx, vy in velocities) / len(velocities)
        confidence = 1.0 / (1.0 + math.sqrt(speed_var))
        dt_pred = max(points[-1].timestamp - points[-2].timestamp, 1.0)
        pred_x = points[-1].x + avg_vx * dt_pred
        pred_y = points[-1].y + avg_vy * dt_pred
        return pred_x / sequence.width, pred_y / sequence.height, confidence

    def status(self) -> str:
        return "heuristic baseline active"
