from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class ViewportMapping:
    case_id: str
    slice_id: int
    image_left: float
    image_top: float
    image_width: float
    image_height: float
    image_pixel_width: int
    image_pixel_height: int
    screen_width: int
    screen_height: int
    timestamp: float


class ViewportService:
    def __init__(self):
        self._latest: Optional[ViewportMapping] = None
        self._by_case_slice: Dict[Tuple[str, int], ViewportMapping] = {}

    def update(self, mapping: ViewportMapping) -> None:
        key = (mapping.case_id, mapping.slice_id)
        self._by_case_slice[key] = mapping
        self._latest = mapping

    def get(self, case_id: str, slice_id: int) -> Optional[ViewportMapping]:
        return self._by_case_slice.get((case_id, slice_id)) or self._latest

    def map_display_to_image(
        self,
        display_x: float,
        display_y: float,
        case_id: str,
        slice_id: int,
    ) -> Optional[Tuple[float, float]]:
        mapping = self.get(case_id, slice_id)
        if not mapping or mapping.image_width <= 0 or mapping.image_height <= 0:
            return None
        rel_x = (display_x - mapping.image_left) / mapping.image_width
        rel_y = (display_y - mapping.image_top) / mapping.image_height
        rel_x = max(0.0, min(1.0, rel_x))
        rel_y = max(0.0, min(1.0, rel_y))
        image_x = rel_x * mapping.image_pixel_width
        image_y = rel_y * mapping.image_pixel_height
        return image_x, image_y
