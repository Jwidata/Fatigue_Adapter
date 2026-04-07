from __future__ import annotations

import math
from typing import List, Optional, Tuple

from app.models.schemas import RoiBBox, RoiShape, RoiStatusResponse
from app.utils.geometry_utils import denormalize_bbox, point_in_bbox, point_in_polygon


def compute_roi_status(
    x: float,
    y: float,
    rois: List[RoiShape],
    width: int,
    height: int,
) -> RoiStatusResponse:
    if not rois:
        return RoiStatusResponse(inside_roi=False, distance_px=None, reason="no_rois")

    inside = False
    nearest_center = None
    nearest_dist = None
    nearest_bbox = None

    for roi in rois:
        if roi.type == "bbox" and roi.bbox:
            bbox = denormalize_bbox((roi.bbox.x, roi.bbox.y, roi.bbox.w, roi.bbox.h), width, height)
            if point_in_bbox(x, y, bbox):
                inside = True
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2
            dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if nearest_dist is None or dist < nearest_dist:
                nearest_dist = dist
                nearest_center = (cx, cy)
                nearest_bbox = roi.bbox
        elif roi.type == "polygon" and roi.polygon:
            points = [[p[0] * width, p[1] * height] for p in roi.polygon.points]
            if point_in_polygon(x, y, points):
                inside = True
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if nearest_dist is None or dist < nearest_dist:
                nearest_dist = dist
                nearest_center = (cx, cy)
                nearest_bbox = RoiBBox(
                    x=min(xs) / width,
                    y=min(ys) / height,
                    w=(max(xs) - min(xs)) / width,
                    h=(max(ys) - min(ys)) / height,
                )

    if nearest_dist is None:
        return RoiStatusResponse(inside_roi=inside, distance_px=None, reason="roi_shape_missing")
    return RoiStatusResponse(
        inside_roi=inside,
        distance_px=float(nearest_dist),
        roi_center_x=nearest_center[0] if nearest_center else None,
        roi_center_y=nearest_center[1] if nearest_center else None,
        roi_bbox=nearest_bbox,
    )
