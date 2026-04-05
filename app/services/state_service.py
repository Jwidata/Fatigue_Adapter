from __future__ import annotations

from typing import Dict, Optional

from app.models.schemas import MetricsResponse, RiskResponse, StateResponse
from app.services.adaptation_outcome_service import AdaptationOutcomeService
from app.services.adaptation_policy_service import AdaptationPolicyService
from app.services.attention_risk_service import AttentionRiskService
from app.services.attention_service import AttentionService
from app.services.catalog_service import CatalogService
from app.services.gaze_manager import GazeManager
from app.services.gaze_prediction_service import GazePredictionService
from app.services.policy_learning_service import PolicyLearningService
from app.services.roi_service import RoiService


class PredictiveStateEngine:
    def __init__(self, config: Dict):
        thresholds = config.get("state_thresholds", {})
        self.low_roi_coverage = thresholds.get("roi_coverage_low", 25.0)

    def classify(self, metrics: MetricsResponse, risk: RiskResponse) -> str:
        if risk.risk_level == "high":
            return "at_risk"
        if metrics.roi_coverage_pct < self.low_roi_coverage and risk.risk_level == "medium":
            return "drifting_attention"
        if metrics.roi_coverage_pct < self.low_roi_coverage and risk.risk_level == "low":
            return "low_roi_engagement"
        return "normal"


class StateService:
    def __init__(
        self,
        catalog: CatalogService,
        gaze: GazeManager,
        roi: RoiService,
        prediction: GazePredictionService,
        risk: AttentionRiskService,
        policy: AdaptationPolicyService,
        outcome: AdaptationOutcomeService,
        learning: PolicyLearningService,
        config: Dict,
    ):
        self.catalog = catalog
        self.gaze = gaze
        self.roi = roi
        self.prediction = prediction
        self.risk = risk
        self.policy = policy
        self.outcome = outcome
        self.learning = learning
        self.config = config
        self.attention = AttentionService(config)
        self.state_engine = PredictiveStateEngine(config)
        self.latest_state: Optional[StateResponse] = None
        self.predictive_enabled = bool(config.get("ui", {}).get("predictive_enabled", True))

        self.case_id, self.series_id = self.catalog.get_default_case()
        self.slice_id = 0

    def reset(self):
        self.latest_state = None
        self.outcome.pending = None

    def set_predictive_enabled(self, enabled: bool):
        self.predictive_enabled = enabled

    def set_active_case(self, case_id: str, slice_id: int | None = None):
        if case_id not in self.catalog._catalog:
            return
        self.case_id = case_id
        self.series_id = self.catalog._catalog[case_id]["default_series"]
        if slice_id is not None:
            self.slice_id = slice_id

    def compute_state(self) -> StateResponse:
        meta = self.catalog.get_slice_meta(self.case_id, self.slice_id)
        self.gaze.set_context(self.case_id, self.slice_id)
        self.gaze.sample_if_needed(meta.width, meta.height)
        points = self.gaze.get_window_points()
        rois = self.roi.get_rois(self.case_id, self.slice_id).rois
        metrics = self.attention.compute_metrics(points, rois, meta.width, meta.height)
        latest_gaze = self.gaze.get_latest()
        self.prediction.update_evaluation(latest_gaze)
        prediction = None
        risk = RiskResponse(risk_level="low", reason="predictive_disabled", predicted_in_roi=False, roi_priority=0.0)
        if self.predictive_enabled:
            prediction = self.prediction.predict(points, meta.width, meta.height)
            risk = self.risk.assess(prediction, rois, meta.width, meta.height)

        state = self.state_engine.classify(metrics, risk)
        adaptation = self.policy.select(state, risk)
        if adaptation.command.actions:
            if not self.outcome.pending or self.outcome.pending.success is not None:
                self.outcome.start(adaptation.command.actions[0])
        outcome = self.outcome.update(latest_gaze, rois, meta.width, meta.height)
        if outcome and outcome.success is not None and outcome.adaptation_id not in self.outcome.logged_ids:
            reward = 1.0 if outcome.success else -1.0
            self.learning.record(state, outcome.action, reward, state)
            self.outcome.logged_ids.add(outcome.adaptation_id)

        response = StateResponse(
            state=state,
            metrics=metrics,
            latest_gaze=latest_gaze,
            window_ms=self.gaze.window_ms,
            prediction=prediction,
            risk=risk,
            attention_status="predicted_in_roi" if risk.predicted_in_roi else "predicted_miss",
        )
        self.latest_state = response
        return response
