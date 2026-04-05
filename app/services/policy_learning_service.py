from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List


@dataclass
class Transition:
    state: str
    action: str
    reward: float
    next_state: str
    timestamp: str


class PolicyLearningService:
    """Lightweight placeholder for RL-ready logging."""

    def __init__(self, config: Dict):
        self.config = config
        self.transitions: List[Transition] = []
        self.log_path = Path("data/policy_transitions.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, state: str, action: str, reward: float, next_state: str):
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            timestamp=datetime.now(UTC).isoformat(),
        )
        self.transitions.append(transition)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json_line(asdict(transition)) + "\n")

    def latest(self) -> Transition | None:
        return self.transitions[-1] if self.transitions else None


def json_line(payload: Dict) -> str:
    import json

    return json.dumps(payload)
