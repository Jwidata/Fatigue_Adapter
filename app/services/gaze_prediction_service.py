from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

from app.models.schemas import GazePoint, PredictionEvalStats, PredictionInfoResponse, PredictionPoint, PredictionResponse
from app.services.feature_builder import FeatureBuilder
from app.services.predictors.predictor_manager import PredictorManager
from app.utils.geometry_utils import clamp_point


class GazePredictionService:
    def __init__(self, config: Dict):
        self.config = config
        pred_cfg = config.get("prediction", {})
        self.min_points = int(pred_cfg.get("min_points", 6))
        self.max_points = int(pred_cfg.get("max_points", 12))
        self.sequence_len = int(pred_cfg.get("sequence_len", 30))
        self.target_mode = pred_cfg.get("target_mode", "delta")
        self.dataset_summary = Path(pred_cfg.get("dataset_summary", "data/gaze_prediction_dataset_summary.json"))
        self.feature_names = pred_cfg.get("feature_names", [])
        if self.dataset_summary.exists():
            summary = json.loads(self.dataset_summary.read_text(encoding="utf-8"))
            self.feature_names = summary.get("feature_names", self.feature_names)
            self.sequence_len = int(summary.get("sequence_len", self.sequence_len))
            self.target_mode = summary.get("target", self.target_mode)
        if not self.feature_names:
            self.feature_names = [
                "x",
                "y",
                "dx",
                "dy",
                "speed",
                "accel",
                "dir_sin",
                "dir_cos",
                "dist_roi",
                "inside_roi",
                "dist_roi_edge",
            ]
        self.feature_builder = FeatureBuilder(self.feature_names)
        if self.max_points < self.sequence_len:
            self.max_points = self.sequence_len
        self.predictor_mode = config.get("prediction", {}).get("predictor_mode", "auto")
        self.log_predictor = bool(config.get("prediction", {}).get("log_predictor", True))
        self.manager = PredictorManager(config)
        self._logged_heuristic = False
        self.last_prediction = None
        self.last_prediction_time = None
        self.last_confidence = None
        self.error_history: List[float] = []
        self.max_errors = 200
        self.model_status = "heuristic"
        self.eval_log_path = Path("data/prediction_eval.jsonl")
        self.eval_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_status = "heuristic"

    def predict(self, points: List[GazePoint], rois, image_width: int, image_height: int) -> PredictionResponse:
        method = "heuristic_velocity"
        timestamp = points[-1].timestamp if points else 0.0
        if len(points) < self.min_points:
            return PredictionResponse(predicted=None, method=method, timestamp=timestamp, based_on=len(points))

        sequence = points[-self.max_points :]
        feature_seq = self.feature_builder.build(
            sequence,
            rois,
            image_width,
            image_height,
            self.sequence_len,
            self.target_mode,
        )
        prediction = self.manager.predict(feature_seq)
        if prediction:
            pred_x_norm, pred_y_norm, confidence = prediction
            pred_x = pred_x_norm * image_width
            pred_y = pred_y_norm * image_height
            pred_x, pred_y = clamp_point(pred_x, pred_y, image_width, image_height)
            method = self.manager.active_name
            self._store_prediction(pred_x, pred_y, timestamp, confidence)
            return PredictionResponse(
                predicted=PredictionPoint(x=pred_x, y=pred_y, confidence=confidence),
                method=method,
                timestamp=timestamp,
                based_on=len(sequence),
            )

        if not self._logged_heuristic:
            self._log("using_heuristic_predictor")
            self._logged_heuristic = True

        recent = points[-self.max_points :]
        velocities = []
        for prev, curr in zip(recent, recent[1:]):
            dt = max(curr.timestamp - prev.timestamp, 1.0)
            vx = (curr.x - prev.x) / dt
            vy = (curr.y - prev.y) / dt
            velocities.append((vx, vy))

        if not velocities:
            return PredictionResponse(predicted=None, method=method, timestamp=timestamp, based_on=len(points))

        avg_vx = sum(v[0] for v in velocities) / len(velocities)
        avg_vy = sum(v[1] for v in velocities) / len(velocities)
        mean_speed = math.sqrt(avg_vx**2 + avg_vy**2)

        speed_var = sum(((math.sqrt(vx**2 + vy**2) - mean_speed) ** 2) for vx, vy in velocities) / len(velocities)
        confidence = 1.0 / (1.0 + math.sqrt(speed_var))

        dt_pred = max(recent[-1].timestamp - recent[-2].timestamp, 1.0)
        pred_x = recent[-1].x + avg_vx * dt_pred
        pred_y = recent[-1].y + avg_vy * dt_pred
        pred_x, pred_y = clamp_point(pred_x, pred_y, image_width, image_height)

        prediction = PredictionPoint(x=pred_x, y=pred_y, confidence=confidence)
        self._store_prediction(pred_x, pred_y, timestamp, confidence)
        return PredictionResponse(predicted=prediction, method=method, timestamp=timestamp, based_on=len(recent))

    def _log(self, message: str):
        if not self.log_predictor:
            return
        print(f"predictor: {message}")

    def update_evaluation(self, actual_point: Optional[GazePoint]):
        if not actual_point or self.last_prediction is None or self.last_prediction_time is None:
            return
        if actual_point.timestamp <= self.last_prediction_time:
            return
        dx = actual_point.x - self.last_prediction[0]
        dy = actual_point.y - self.last_prediction[1]
        error = math.sqrt(dx * dx + dy * dy)
        self.error_history.append(error)
        if len(self.error_history) > self.max_errors:
            self.error_history.pop(0)
        self._log_eval(actual_point, error)

    def get_eval_stats(self) -> PredictionEvalStats:
        if not self.error_history:
            return PredictionEvalStats(
                count=0,
                mean_error_px=None,
                median_error_px=None,
                within_25_px_pct=None,
                within_50_px_pct=None,
            )
        errors = sorted(self.error_history)
        count = len(errors)
        mean = sum(errors) / count
        median = errors[count // 2] if count % 2 == 1 else (errors[count // 2 - 1] + errors[count // 2]) / 2
        within_25 = len([e for e in errors if e <= 25]) / count * 100
        within_50 = len([e for e in errors if e <= 50]) / count * 100
        return PredictionEvalStats(
            count=count,
            mean_error_px=mean,
            median_error_px=median,
            within_25_px_pct=within_25,
            within_50_px_pct=within_50,
        )

    def get_info(self) -> PredictionInfoResponse:
        active = self.manager.active_name
        explanation_map = {
            "heuristic": "Velocity-based baseline using recent gaze deltas.",
            "constant_velocity": "Constant velocity baseline for next-step gaze.",
            "xgboost": "Tree-based model using gaze trajectory features.",
            "gru": "Sequence model capturing temporal gaze dynamics.",
            "lstm": "Sequence model with long-range temporal memory.",
            "transformer": "Attention-based model for complex gaze patterns.",
            "temporal_cnn": "Temporal convolution over short-range gaze motion.",
        }
        explanation = explanation_map.get(active, "Gaze prediction model.")
        accuracy_note = (
            "Prediction accuracy not yet evaluated in this prototype"
            if len(self.error_history) < 10
            else "Prediction accuracy shown in debug metrics"
        )
        status_map = self.manager.status()
        model_status = status_map.get(active, "heuristic baseline active")
        if active == "heuristic" and self.manager.fallback_reason:
            model_status = self.manager.fallback_reason
        return PredictionInfoResponse(
            predictor_mode=self.predictor_mode,
            active_predictor=active,
            explanation=explanation,
            model_status=model_status,
            confidence=self.last_confidence,
            accuracy_note=accuracy_note,
            eval_stats=self.get_eval_stats(),
        )

    def _store_prediction(self, x: float, y: float, timestamp: float, confidence: float):
        self.last_prediction = (x, y)
        self.last_prediction_time = timestamp
        self.last_confidence = confidence

    def _log_eval(self, actual_point: GazePoint, error: float):
        payload = {
            "timestamp": actual_point.timestamp,
            "pred_x": self.last_prediction[0],
            "pred_y": self.last_prediction[1],
            "actual_x": actual_point.x,
            "actual_y": actual_point.y,
            "error_px": error,
        }
        with self.eval_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json_line(payload) + "\n")


def json_line(payload: Dict) -> str:
    import json

    return json.dumps(payload)
