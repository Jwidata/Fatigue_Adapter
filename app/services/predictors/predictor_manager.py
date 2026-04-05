from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.models.schemas import GazePoint
from app.services.predictors.base_predictor import BasePredictor
from app.services.predictors.constant_velocity_predictor import ConstantVelocityPredictor
from app.services.predictors.cnn_predictor import TemporalCNNPredictor
from app.services.predictors.gru_predictor import GRUPredictor
from app.services.predictors.heuristic_predictor import HeuristicPredictor
from app.services.predictors.lstm_predictor import LSTMPredictor
from app.services.predictors.transformer_predictor import TransformerPredictor
from app.services.predictors.xgboost_predictor import XGBoostPredictor


class PredictorManager:
    def __init__(self, config: Dict):
        pred_cfg = config.get("prediction", {})
        self.mode = pred_cfg.get("predictor_mode", "auto")
        self.log_predictor = bool(pred_cfg.get("log_predictor", True))
        self.lstm_weights = Path(pred_cfg.get("lstm_weights", "configs/lstm_gaze_weights.pt"))
        self.transformer_weights = Path(pred_cfg.get("transformer_weights", "configs/transformer_gaze_weights.pt"))
        self.gru_weights = Path(pred_cfg.get("gru_weights", "configs/gru_gaze_weights.pt"))
        self.cnn_weights = Path(pred_cfg.get("cnn_weights", "configs/cnn_gaze_weights.pt"))
        self.xgb_x = Path(pred_cfg.get("xgboost_model_x", "configs/xgboost_gaze_model_x.json"))
        self.xgb_y = Path(pred_cfg.get("xgboost_model_y", "configs/xgboost_gaze_model_y.json"))

        self.heuristic = HeuristicPredictor()
        self.constant_velocity = ConstantVelocityPredictor()
        self.lstm = LSTMPredictor(self.lstm_weights)
        self.gru = GRUPredictor(self.gru_weights)
        self.cnn = TemporalCNNPredictor(self.cnn_weights)
        self.transformer = TransformerPredictor(self.transformer_weights)
        self.xgboost = XGBoostPredictor(self.xgb_x, self.xgb_y)
        self.active_name = "heuristic"
        self.fallback_reason = ""

    def available_predictors(self) -> Dict[str, bool]:
        return {
            "heuristic": self.heuristic.available(),
            "constant_velocity": self.constant_velocity.available(),
            "lstm": self.lstm.available(),
            "gru": self.gru.available(),
            "temporal_cnn": self.cnn.available(),
            "transformer": self.transformer.available(),
            "xgboost": self.xgboost.available(),
        }

    def select(self) -> BasePredictor:
        self.fallback_reason = ""
        if self.mode == "heuristic":
            self.active_name = "heuristic"
            return self.heuristic
        if self.mode == "constant_velocity":
            self.active_name = "constant_velocity"
            return self.constant_velocity
        if self.mode == "lstm" and self.lstm.available():
            self.active_name = "lstm"
            return self.lstm
        if self.mode == "gru" and self.gru.available():
            self.active_name = "gru"
            return self.gru
        if self.mode == "temporal_cnn" and self.cnn.available():
            self.active_name = "temporal_cnn"
            return self.cnn
        if self.mode == "transformer" and self.transformer.available():
            self.active_name = "transformer"
            return self.transformer
        if self.mode == "xgboost" and self.xgboost.available():
            self.active_name = "xgboost"
            return self.xgboost
        if self.mode == "lstm" and not self.lstm.available():
            self.fallback_reason = "lstm_weights_missing_fallback"
        if self.mode == "gru" and not self.gru.available():
            self.fallback_reason = "gru_weights_missing_fallback"
        if self.mode == "temporal_cnn" and not self.cnn.available():
            self.fallback_reason = "cnn_weights_missing_fallback"
        if self.mode == "transformer" and not self.transformer.available():
            self.fallback_reason = "transformer_weights_missing_fallback"
        if self.mode == "xgboost" and not self.xgboost.available():
            self.fallback_reason = "xgboost_model_missing_fallback"

        if self.mode == "auto":
            if self.transformer.available():
                self.active_name = "transformer"
                return self.transformer
            if self.lstm.available():
                self.active_name = "lstm"
                return self.lstm
            if self.gru.available():
                self.active_name = "gru"
                return self.gru
            if self.cnn.available():
                self.active_name = "temporal_cnn"
                return self.cnn
            if self.xgboost.available():
                self.active_name = "xgboost"
                return self.xgboost
            self.fallback_reason = "auto_fallback_to_heuristic"

        self.active_name = "heuristic"
        return self.heuristic

    def predict(self, sequence: List[GazePoint]) -> Optional[Tuple[float, float, float]]:
        predictor = self.select()
        prediction = predictor.predict(sequence)
        if prediction is None and predictor.name != "heuristic":
            self.active_name = "heuristic"
            prediction = self.heuristic.predict(sequence)
        return prediction

    def status(self) -> Dict[str, str]:
        return {
            "heuristic": self.heuristic.status(),
            "constant_velocity": self.constant_velocity.status(),
            "lstm": self.lstm.status(),
            "gru": self.gru.status(),
            "temporal_cnn": self.cnn.status(),
            "transformer": self.transformer.status(),
            "xgboost": self.xgboost.status(),
        }
