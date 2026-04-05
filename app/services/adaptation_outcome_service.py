from __future__ import annotations

import time
import uuid
from typing import Dict, Optional

from app.models.schemas import AdaptationOutcome, GazePoint, RoiShape
from app.utils.geometry_utils import denormalize_bbox, point_in_bbox, point_in_polygon


class AdaptationOutcomeService:
    def __init__(self, config: Dict):
        self.config = config
        self.window_ms = int(config.get("outcome", {}).get("window_ms", 2000))
        self.pending: Optional[AdaptationOutcome] = None
        self.logged_ids: set[str] = set()

    def start(self, action: str):
        self.pending = AdaptationOutcome(
            adaptation_id=str(uuid.uuid4()),
            action=action,
            success=None,
            time_to_roi_ms=None,
            notes=None,
            timestamp=time.time() * 1000,
        )

    def update(self, gaze: Optional[GazePoint], rois: list[RoiShape], image_width: int, image_height: int) -> Optional[AdaptationOutcome]:
        if not self.pending:
            return None
        now = time.time() * 1000
        if gaze and self._point_in_rois(gaze.x, gaze.y, rois, image_width, image_height):
            self.pending.success = True
            self.pending.time_to_roi_ms = now - self.pending.timestamp
            return self.pending
        if now - self.pending.timestamp > self.window_ms:
            self.pending.success = False
            self.pending.time_to_roi_ms = None
            self.pending.notes = "no_roi_fixation_in_window"
            return self.pending
        return self.pending

    def get_latest(self) -> Optional[AdaptationOutcome]:
        return self.pending

    @staticmethod
    def _point_in_rois(x: float, y: float, rois: list[RoiShape], width: int, height: int) -> bool:
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
