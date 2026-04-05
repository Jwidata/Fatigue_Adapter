from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

from app.adapters.gaze.base import GazeAdapter
from app.models.schemas import GazePoint


class ReplayGazeAdapter(GazeAdapter):
    def __init__(self, replay_path: Path | None = None):
        self.replay_path = replay_path
        self._points: List[GazePoint] = []
        self._index = 0
        if replay_path and replay_path.exists():
            self._load(replay_path)

    def _load(self, path: Path):
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    self._points.append(
                        GazePoint(
                            timestamp=float(row.get("timestamp", 0.0)),
                            x=float(row.get("x", 0.0)),
                            y=float(row.get("y", 0.0)),
                            source="replay",
                            mode=row.get("mode") or "replay",
                            pupil=float(row.get("pupil")) if row.get("pupil") else None,
                            blink=row.get("blink") == "true",
                        )
                    )
                except ValueError:
                    continue

    def next_point(self, image_width: int, image_height: int, timestamp: float) -> Optional[GazePoint]:
        if not self._points:
            return None
        point = self._points[self._index]
        self._index = (self._index + 1) % len(self._points)
        return point
