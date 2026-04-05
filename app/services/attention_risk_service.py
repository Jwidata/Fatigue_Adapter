from __future__ import annotations

from typing import Dict, List

from app.models.schemas import PredictionResponse, RiskResponse, RoiShape
from app.utils.geometry_utils import denormalize_bbox, point_in_bbox, point_in_polygon


class AttentionRiskService:
    def __init__(self, config: Dict):
        self.config = config
        self.high_priority = float(config.get("risk", {}).get("high_priority", 0.7))
        self.min_priority = float(config.get("risk", {}).get("min_roi_priority", 0.6))
        self.high_confidence = float(config.get("risk", {}).get("high_confidence", 0.5))

    def assess(
        self,
        prediction: PredictionResponse,
        rois: List[RoiShape],
        image_width: int,
        image_height: int,
    ) -> RiskResponse:
        if prediction.predicted is None or not rois:
            return RiskResponse(
                risk_level="low",
                reason="no_prediction_or_roi",
                predicted_in_roi=False,
                roi_priority=0.0,
            )

        roi_priority = max((roi.priority for roi in rois), default=0.0)
        predicted_in_roi = self._point_in_rois(prediction.predicted.x, prediction.predicted.y, rois, image_width, image_height)

        if predicted_in_roi:
            return RiskResponse(
                risk_level="low",
                reason="predicted_in_roi",
                predicted_in_roi=True,
                roi_priority=roi_priority,
            )
        if roi_priority >= self.min_priority and prediction.predicted.confidence >= self.high_confidence:
            return RiskResponse(
                risk_level="high",
                reason="predicted_miss_of_critical_roi",
                predicted_in_roi=False,
                roi_priority=roi_priority,
            )
        if roi_priority >= self.min_priority:
            return RiskResponse(
                risk_level="medium",
                reason="predicted_miss_of_roi",
                predicted_in_roi=False,
                roi_priority=roi_priority,
            )
        return RiskResponse(
            risk_level="low",
            reason="predicted_miss_low_priority",
            predicted_in_roi=False,
            roi_priority=roi_priority,
        )

    @staticmethod
    def _point_in_rois(x: float, y: float, rois: List[RoiShape], width: int, height: int) -> bool:
        for roi in rois:
            if roi.type == "bbox" and roi.bbox:
                bbox = denormalize_bbox((roi.bbox.x, roi.bbox.y, roi.bbox.w, roi.bbox.h), width, height)
                if point_in_bbox(x, y, bbox):
                    return True
            if roi.type == "polygon" and roi.polygon:
                poly = [[p[0] * width, p[1] * height] for p in roi.polygon.points]
                if point_in_polygon(x, y, poly):
                    return True
        return False
