from __future__ import annotations

import math
from typing import Dict, List, Optional

from app.models.schemas import GazePoint, MetricsResponse, RoiShape
from app.utils.geometry_utils import denormalize_bbox, point_in_bbox, point_in_polygon


class AttentionService:
    def __init__(self, config: Dict):
        self.config = config

    def compute_metrics(
        self,
        points: List[GazePoint],
        rois: List[RoiShape],
        image_width: int,
        image_height: int,
    ) -> MetricsResponse:
        if not points:
            return MetricsResponse(
                dwell_time_ms=0.0,
                roi_hits=0,
                first_fixation_delay_ms=None,
                roi_coverage_pct=0.0,
                dispersion=0.0,
                scan_coverage_pct=0.0,
            )

        roi_hits = 0
        inside_samples = 0
        first_fixation_delay_ms: Optional[float] = None
        last_inside = False
        timestamps = [p.timestamp for p in points]
        window_start = min(timestamps)

        for point in points:
            inside = self._point_in_rois(point, rois, image_width, image_height)
            if inside:
                inside_samples += 1
                if not last_inside:
                    roi_hits += 1
                    if first_fixation_delay_ms is None:
                        first_fixation_delay_ms = point.timestamp - window_start
            last_inside = inside

        sample_count = len(points)
        if sample_count > 1:
            total_time = max(timestamps) - min(timestamps)
            avg_interval = total_time / (sample_count - 1)
        else:
            avg_interval = 0.0
        dwell_time_ms = inside_samples * avg_interval
        roi_coverage_pct = (inside_samples / sample_count) * 100

        dispersion = self._dispersion(points, image_width, image_height)
        scan_coverage_pct = self._scan_coverage(points, image_width, image_height)

        return MetricsResponse(
            dwell_time_ms=dwell_time_ms,
            roi_hits=roi_hits,
            first_fixation_delay_ms=first_fixation_delay_ms,
            roi_coverage_pct=roi_coverage_pct,
            dispersion=dispersion,
            scan_coverage_pct=scan_coverage_pct,
        )

    def _point_in_rois(
        self,
        point: GazePoint,
        rois: List[RoiShape],
        width: int,
        height: int,
    ) -> bool:
        for roi in rois:
            if roi.type == "bbox" and roi.bbox:
                bbox = denormalize_bbox((roi.bbox.x, roi.bbox.y, roi.bbox.w, roi.bbox.h), width, height)
                if point_in_bbox(point.x, point.y, bbox):
                    return True
            if roi.type == "polygon" and roi.polygon:
                poly = [[p[0] * width, p[1] * height] for p in roi.polygon.points]
                if point_in_polygon(point.x, point.y, poly):
                    return True
        return False

    @staticmethod
    def _dispersion(points: List[GazePoint], width: int, height: int) -> float:
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        var_x = sum((x - mean_x) ** 2 for x in xs) / len(xs)
        var_y = sum((y - mean_y) ** 2 for y in ys) / len(ys)
        std = math.sqrt(var_x + var_y)
        diag = math.sqrt(width ** 2 + height ** 2) or 1.0
        return std / diag

    @staticmethod
    def _scan_coverage(points: List[GazePoint], width: int, height: int) -> float:
        grid = 10
        visited = set()
        for point in points:
            gx = min(grid - 1, int((point.x / max(width, 1)) * grid))
            gy = min(grid - 1, int((point.y / max(height, 1)) * grid))
            visited.add((gx, gy))
        return (len(visited) / (grid * grid)) * 100
