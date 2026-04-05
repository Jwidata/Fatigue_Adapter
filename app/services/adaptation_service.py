from __future__ import annotations

import time
from typing import Dict

from app.models.schemas import AdaptationCommand, AdaptationResponse
from app.services.state_service import StateService


class AdaptationService:
    def __init__(self, config: Dict, state_service: StateService):
        self.config = config
        self.state_service = state_service
        self._latest = None

    def get_latest(self) -> AdaptationResponse:
        if self._latest:
            return self._latest
        return self._build_from_state(self.state_service.compute_state())

    def update_from_state(self, state):
        return self._build_from_state(state)

    def _build_from_state(self, state):
        if state.state == "overload":
            command = AdaptationCommand(
                state=state.state,
                actions=["highlight_roi", "dim_side_info"],
                highlight_strength=0.7,
                dim_level=0.4,
                message=None,
            )
        elif state.state == "fatigue":
            command = AdaptationCommand(
                state=state.state,
                actions=["highlight_roi", "show_review_prompt", "suggest_zoom"],
                highlight_strength=1.0,
                dim_level=0.5,
                message="Review this area",
            )
        else:
            command = AdaptationCommand(
                state=state.state,
                actions=[],
                highlight_strength=0.2,
                dim_level=0.0,
                message=None,
            )
        response = AdaptationResponse(command=command, timestamp=time.time() * 1000, metrics=state.metrics)
        self._latest = response
        return response
