from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

from app.adapters.gaze.replay import ReplayGazeAdapter
from app.adapters.gaze.synthetic import SyntheticGazeAdapter
from app.adapters.gaze.tobii import TobiiGazeAdapter
from app.models.schemas import GazePoint
from app.services.catalog_service import CatalogService


class GazeManager:
    def __init__(self, catalog: CatalogService, config: Dict):
        self.catalog = catalog
        self.config = config
        self.max_buffer = int(config.get("gaze", {}).get("buffer_size", 300))
        self.buffer: Deque[GazePoint] = deque(maxlen=self.max_buffer)
        self.window_ms = int(config.get("attention", {}).get("window_ms", 4000))
        self.mode = config.get("gaze", {}).get("default_mode", "normal")
        self.source = config.get("gaze", {}).get("default_source", "synthetic")
        replay_path = Path(__file__).resolve().parents[2] / "configs" / "replay_gaze.csv"
        self.synthetic = SyntheticGazeAdapter(self.mode)
        self.replay = ReplayGazeAdapter(replay_path if replay_path.exists() else None)
        self.tobii = TobiiGazeAdapter()
        self.case_id = None
        self.slice_id = None

    def reset(self):
        self.buffer = deque(maxlen=self.max_buffer)

    def set_mode(self, mode: str):
        self.mode = mode
        self.synthetic.set_mode(mode)

    def set_source(self, source: str):
        self.source = source

    def add_point(self, point: GazePoint):
        self.buffer.append(point)
        self._log_point(point)

    def set_context(self, case_id: str, slice_id: int):
        self.case_id = case_id
        self.slice_id = slice_id

    def get_latest(self) -> Optional[GazePoint]:
        return self.buffer[-1] if self.buffer else None

    def get_window_points(self) -> List[GazePoint]:
        now = time.time() * 1000
        cutoff = now - self.window_ms
        return [p for p in self.buffer if p.timestamp >= cutoff]

    def get_recent_points(self, count: int) -> List[GazePoint]:
        if count <= 0:
            return []
        return list(self.buffer)[-count:]

    def sample_if_needed(self, image_width: int, image_height: int) -> Optional[GazePoint]:
        timestamp = time.time() * 1000
        if self.source == "synthetic":
            point = self.synthetic.next_point(image_width, image_height, timestamp)
        elif self.source == "replay":
            point = self.replay.next_point(image_width, image_height, timestamp)
            if point:
                point.timestamp = timestamp
        elif self.source == "tobii":
            point = self.tobii.next_point(image_width, image_height, timestamp)
            if point is None:
                point = self.synthetic.next_point(image_width, image_height, timestamp)
        else:
            point = None
        if point:
            point.mode = self.mode
            self.add_point(point)
        return point

    def _log_point(self, point: GazePoint):
        log_path = Path(__file__).resolve().parents[2] / "data" / "gaze_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": point.timestamp,
            "x": point.x,
            "y": point.y,
            "source": point.source,
            "mode": point.mode,
            "case_id": self.case_id,
            "slice_id": self.slice_id,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json_line(payload) + "\n")


def json_line(payload: Dict) -> str:
    import json

    return json.dumps(payload)
