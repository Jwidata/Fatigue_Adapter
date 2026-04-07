from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

try:
    import xgboost as xgb
except Exception:
    xgb = None

from app.services.feature_builder import FeatureSequence
from app.services.predictors.base_predictor import BasePredictor
from app.services.predictors.predictor_utils import reconstruct_absolute


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

    def predict(self, sequence: FeatureSequence) -> Optional[Tuple[float, float, float]]:
        if not self.loaded or self.model_x is None or self.model_y is None:
            return None
        flat = sequence.features.reshape(1, -1)
        pred_x = self.model_x.predict(flat)[0]
        pred_y = self.model_y.predict(flat)[0]
        norm_x, norm_y = reconstruct_absolute(
            float(pred_x),
            float(pred_y),
            sequence.last_pos_norm,
            sequence.last_delta_norm,
            sequence.target_mode,
        )
        return float(norm_x), float(norm_y), 0.5

    def status(self) -> str:
        if xgb is None:
            return "missing_dependency"
        return "weights loaded" if self.loaded else "weights missing"
