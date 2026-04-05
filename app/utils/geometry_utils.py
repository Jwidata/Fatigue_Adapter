from __future__ import annotations

from typing import Iterable, List, Tuple


def point_in_bbox(x: float, y: float, bbox: Tuple[float, float, float, float]) -> bool:
    bx, by, bw, bh = bbox
    return bx <= x <= bx + bw and by <= y <= by + bh


def point_in_polygon(x: float, y: float, points: List[List[float]]) -> bool:
    inside = False
    n = len(points)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = points[i]
        xj, yj = points[j]
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_bbox(bbox: Tuple[float, float, float, float], width: int, height: int) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    return (x / width, y / height, w / width, h / height)


def denormalize_bbox(bbox: Tuple[float, float, float, float], width: int, height: int) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    return (x * width, y * height, w * width, h * height)


def clamp_point(x: float, y: float, width: int, height: int) -> Tuple[float, float]:
    return (clamp(x, 0, width - 1), clamp(y, 0, height - 1))
