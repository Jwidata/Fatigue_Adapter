from __future__ import annotations

import time
from typing import Dict

from app.models.schemas import AdaptationCommand, AdaptationResponse, RiskResponse


class AdaptationPolicyService:
    def __init__(self, config: Dict):
        self.config = config
        self._latest = None
        self.adaptive_enabled = True
        self.policy_mode = config.get("ui", {}).get("policy_mode", "balanced")
        self.min_roi_priority = float(config.get("risk", {}).get("min_roi_priority", 0.6))
        policy_cfg = config.get("policy", {})
        self.message_min_risk = policy_cfg.get("message_min_risk", "medium")
        self.intervention_min_risk = policy_cfg.get("intervention_min_risk", "high")
        self.cooldown_ms = int(policy_cfg.get("cooldown_ms", 2500))
        self.message_hold_ms = int(policy_cfg.get("message_hold_ms", 1200))
        self.adaptation_hold_ms = int(policy_cfg.get("adaptation_hold_ms", 800))
        self.last_intervention_ts = None
        self.silent_low_risk = bool(config.get("ui", {}).get("silent_low_risk", True))
        self._last_command = None
        self._last_command_ts = None
        if self.policy_mode:
            self.set_policy_mode(self.policy_mode)

    def set_adaptive_enabled(self, enabled: bool):
        self.adaptive_enabled = enabled

    def set_policy_mode(self, mode: str):
        if mode == "conservative":
            mode = "quiet"
        self.policy_mode = mode
        if mode == "quiet":
            self.message_min_risk = "high"
            self.intervention_min_risk = "high"
        elif mode == "aggressive":
            self.message_min_risk = "medium"
            self.intervention_min_risk = "medium"
        else:
            self.message_min_risk = "medium"
            self.intervention_min_risk = "high"

    def update_policy_config(
        self,
        message_min_risk: str | None = None,
        intervention_min_risk: str | None = None,
        cooldown_ms: int | None = None,
        message_hold_ms: int | None = None,
        silent_low_risk: bool | None = None,
    ):
        if message_min_risk:
            self.message_min_risk = message_min_risk
        if intervention_min_risk:
            self.intervention_min_risk = intervention_min_risk
        if cooldown_ms is not None:
            self.cooldown_ms = int(cooldown_ms)
        if message_hold_ms is not None:
            self.message_hold_ms = int(message_hold_ms)
        if silent_low_risk is not None:
            self.silent_low_risk = bool(silent_low_risk)

    def select(self, state: str, risk: RiskResponse) -> AdaptationResponse:
        if not self.adaptive_enabled:
            command = AdaptationCommand(
                state=state,
                actions=[],
                highlight_strength=0.0,
                dim_level=0.0,
                message=None,
            )
            return self._wrap(command)

        if self._in_cooldown():
            return self._wrap(
                AdaptationCommand(state=state, actions=[], highlight_strength=0.0, dim_level=0.0, message=None)
            )

        should_act = self._risk_at_least(risk.risk_level, self.intervention_min_risk)
        if should_act and risk.roi_priority < self.min_roi_priority:
            should_act = False

        if should_act and risk.risk_level == "high":
            command = AdaptationCommand(
                state=state,
                actions=["strong_highlight_roi", "dim_non_roi", "show_focus_prompt", "recommend_zoom"],
                highlight_strength=1.0,
                dim_level=0.6,
                message="Predicted attention drift detected — review highlighted region",
            )
        elif should_act and risk.risk_level == "medium":
            command = AdaptationCommand(
                state=state,
                actions=["highlight_roi"],
                highlight_strength=0.7,
                dim_level=0.2,
                message=None,
            )
        else:
            command = AdaptationCommand(
                state=state,
                actions=[],
                highlight_strength=0.2,
                dim_level=0.0,
                message=None,
            )
        response = self._wrap(command)
        if command.actions:
            self.last_intervention_ts = response.timestamp
        self._last_command = command
        self._last_command_ts = response.timestamp
        return response

    def get_latest(self) -> AdaptationResponse:
        return self._latest

    def get_policy_info(self):
        from app.models.schemas import PolicyInfoResponse

        return PolicyInfoResponse(
            policy_mode=self.policy_mode,
            message_min_risk=self.message_min_risk,
            intervention_min_risk=self.intervention_min_risk,
            cooldown_ms=self.cooldown_ms,
            message_hold_ms=self.message_hold_ms,
            last_intervention_ts=self.last_intervention_ts,
            silent_low_risk=self.silent_low_risk,
        )

    def _wrap(self, command: AdaptationCommand) -> AdaptationResponse:
        now = time.time() * 1000
        if self._last_command and self._last_command.actions and not command.actions:
            if self._last_command_ts and (now - self._last_command_ts) < self.adaptation_hold_ms:
                command = self._last_command
        response = AdaptationResponse(command=command, timestamp=now, metrics=None)
        self._latest = response
        return response

    def _in_cooldown(self) -> bool:
        if self.last_intervention_ts is None:
            return False
        return (time.time() * 1000 - self.last_intervention_ts) < self.cooldown_ms

    @staticmethod
    def _risk_at_least(risk_level: str, threshold: str) -> bool:
        order = {"low": 0, "medium": 1, "high": 2}
        return order.get(risk_level, 0) >= order.get(threshold, 1)
