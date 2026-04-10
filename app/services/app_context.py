from __future__ import annotations

from pathlib import Path

from app.services.adaptation_outcome_service import AdaptationOutcomeService
from app.services.adaptation_policy_service import AdaptationPolicyService
from app.services.attention_risk_service import AttentionRiskService
from app.services.catalog_service import CatalogService
from app.services.gaze_manager import GazeManager
from app.services.gaze_prediction_service import GazePredictionService
from app.services.image_service import ImageService
from app.services.policy_learning_service import PolicyLearningService
from app.services.roi_service import RoiService
from app.services.state_service import StateService
from app.services.viewport_service import ViewportService
from app.utils.config_utils import load_config


class AppContext:
    def __init__(self):
        self.config = load_config()
        self.catalog = CatalogService(self.config).load_or_build()
        self.images = ImageService(self.catalog)
        self.roi = RoiService(self.catalog, self.config)
        self.gaze = GazeManager(self.catalog, self.config)
        self.prediction = GazePredictionService(self.config)
        self.risk = AttentionRiskService(self.config)
        self.policy = AdaptationPolicyService(self.config)
        self.policy.set_adaptive_enabled(bool(self.config.get("ui", {}).get("adaptive_enabled", True)))
        self.policy.set_policy_mode(self.config.get("ui", {}).get("policy_mode", "balanced"))
        self.outcome = AdaptationOutcomeService(self.config)
        self.learning = PolicyLearningService(self.config)
        self.viewport = ViewportService()
        self.state_service = StateService(
            self.catalog,
            self.gaze,
            self.roi,
            self.prediction,
            self.risk,
            self.policy,
            self.outcome,
            self.learning,
            self.config,
        )

    @property
    def data_root(self) -> Path:
        return Path(__file__).resolve().parents[2] / "Data"
