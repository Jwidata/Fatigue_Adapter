from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.models.schemas import GazePoint, RoiShape
from app.utils.geometry_utils import point_in_bbox, point_in_polygon


@dataclass
class FeatureSequence:
    features: np.ndarray
    feature_names: List[str]
    last_pos_norm: Tuple[float, float]
    last_delta_norm: Tuple[float, float]
    width: int
    height: int
    target_mode: str
    points: List[GazePoint]


class FeatureBuilder:
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def build(
        self,
        points: List[GazePoint],
        rois: List[RoiShape],
        width: int,
        height: int,
        sequence_len: int,
        target_mode: str,
    ) -> FeatureSequence:
        if not points:
            raise ValueError("no_gaze_points")
        seq = points[-sequence_len:]
        if len(seq) < sequence_len:
            pad_point = seq[0]
            seq = [pad_point] * (sequence_len - len(seq)) + seq

        roi_info = _build_roi_info(rois)
        features: List[List[float]] = []
        prev_dx = 0.0
        prev_dy = 0.0
        for idx, point in enumerate(seq):
            x_norm = point.x / width
            y_norm = point.y / height
            if idx == 0:
                dx = 0.0
                dy = 0.0
            else:
                prev = seq[idx - 1]
                dx = (point.x - prev.x) / width
                dy = (point.y - prev.y) / height
            speed = math.sqrt(dx * dx + dy * dy)
            if idx >= 2:
                accel = math.sqrt((dx - prev_dx) ** 2 + (dy - prev_dy) ** 2)
            else:
                accel = 0.0
            angle = math.atan2(dy, dx) if speed > 0 else 0.0
            dir_sin = math.sin(angle)
            dir_cos = math.cos(angle)

            dist_roi = 0.0
            inside_roi = 0.0
            dist_roi_edge = 0.0
            if roi_info:
                dist_roi = _nearest_roi_center_distance(x_norm, y_norm, roi_info)
                inside_roi = 1.0 if _point_in_rois_norm(x_norm, y_norm, roi_info) else 0.0
                dist_roi_edge = _distance_to_roi_edge(x_norm, y_norm, roi_info)

            row = []
            for name in self.feature_names:
                if name == "x":
                    row.append(x_norm)
                elif name == "y":
                    row.append(y_norm)
                elif name == "dx":
                    row.append(dx)
                elif name == "dy":
                    row.append(dy)
                elif name == "speed":
                    row.append(speed)
                elif name == "accel":
                    row.append(accel)
                elif name == "dir_sin":
                    row.append(dir_sin)
                elif name == "dir_cos":
                    row.append(dir_cos)
                elif name == "dist_roi":
                    row.append(dist_roi)
                elif name == "inside_roi":
                    row.append(inside_roi)
                elif name == "dist_roi_edge":
                    row.append(dist_roi_edge)
                else:
                    row.append(0.0)
            features.append(row)
            prev_dx = dx
            prev_dy = dy

        last = seq[-1]
        last_pos_norm = (last.x / width, last.y / height)
        if len(seq) >= 2:
            prev = seq[-2]
            last_delta_norm = ((last.x - prev.x) / width, (last.y - prev.y) / height)
        else:
            last_delta_norm = (0.0, 0.0)
        return FeatureSequence(
            features=np.array(features, dtype=np.float32),
            feature_names=self.feature_names,
            last_pos_norm=last_pos_norm,
            last_delta_norm=last_delta_norm,
            width=width,
            height=height,
            target_mode=target_mode,
            points=seq,
        )


def _build_roi_info(rois: List[RoiShape]) -> List[Dict]:
    info: List[Dict] = []
    for roi in rois:
        if roi.bbox:
            info.append(
                {
                    "type": "bbox",
                    "bbox": (roi.bbox.x, roi.bbox.y, roi.bbox.w, roi.bbox.h),
                    "center": (roi.bbox.x + roi.bbox.w / 2, roi.bbox.y + roi.bbox.h / 2),
                }
            )
        elif roi.polygon:
            points = roi.polygon.points
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
            center = (sum(xs) / len(xs), sum(ys) / len(ys))
            info.append({"type": "polygon", "points": points, "bbox": bbox, "center": center})
    return info


def _point_in_rois_norm(x: float, y: float, rois: List[Dict]) -> bool:
    for roi in rois:
        if roi["type"] == "bbox" and point_in_bbox(x, y, roi["bbox"]):
            return True
        if roi["type"] == "polygon" and point_in_polygon(x, y, roi["points"]):
            return True
    return False


def _nearest_roi_center_distance(x: float, y: float, rois: List[Dict]) -> float:
    min_dist = None
    for roi in rois:
        cx, cy = roi["center"]
        dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        if min_dist is None or dist < min_dist:
            min_dist = dist
    return min_dist or 0.0


def _distance_to_roi_edge(x: float, y: float, rois: List[Dict]) -> float:
    min_dist = None
    for roi in rois:
        bx, by, bw, bh = roi["bbox"]
        if bx <= x <= bx + bw and by <= y <= by + bh:
            dist = min(abs(x - bx), abs(x - (bx + bw)), abs(y - by), abs(y - (by + bh)))
        else:
            dx = max(bx - x, 0.0, x - (bx + bw))
            dy = max(by - y, 0.0, y - (by + bh))
            dist = math.sqrt(dx * dx + dy * dy)
        if min_dist is None or dist < min_dist:
            min_dist = dist
    return min_dist or 0.0
