from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

try:
    import xgboost as xgb
except Exception:
    xgb = None

from app.models.schemas import GazePoint
from app.services.predictors.base_predictor import BasePredictor


class XGBoostPredictor(BasePredictor):
    name = "xgboost"

    def __init__(self, model_x_path: Path, model_y_path: Path):
        self.model_x_path = model_x_path
        self.model_y_path = model_y_path
        self.loaded = False
        self.model_x = None
        self.model_y = None
        if xgb is None:
            return
        if model_x_path.exists() and model_y_path.exists():
            self.model_x = xgb.XGBRegressor()
            self.model_y = xgb.XGBRegressor()
            self.model_x.load_model(model_x_path)
            self.model_y.load_model(model_y_path)
            self.loaded = True

    def available(self) -> bool:
        return self.loaded

    def predict(self, sequence: List[GazePoint]) -> Optional[Tuple[float, float, float]]:
        if not self.loaded or self.model_x is None or self.model_y is None:
            return None
        flat = [coord for point in sequence for coord in (point.x, point.y)]
        pred_x = self.model_x.predict([flat])[0]
        pred_y = self.model_y.predict([flat])[0]
        return float(pred_x), float(pred_y), 0.5

    def status(self) -> str:
        if xgb is None:
            return "missing_dependency"
        return "weights loaded" if self.loaded else "weights missing"
