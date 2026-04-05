from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None

from app.models.schemas import GazePoint
from app.services.predictors.base_predictor import BasePredictor


if nn is not None:
    class TemporalCNN(nn.Module):
        def __init__(self, input_size: int = 6):
            super().__init__()
            self.conv1 = nn.Conv1d(input_size, 16, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(16, 2)

        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)
            return self.fc(x)
else:
    TemporalCNN = None


class TemporalCNNPredictor(BasePredictor):
    name = "temporal_cnn"

    def __init__(self, weights_path: Path):
        self.weights_path = weights_path
        self.loaded = False
        self.model = None
        if torch is None or nn is None or TemporalCNN is None:
            return
        if weights_path.exists():
            self.model = TemporalCNN()
            state = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state)
            self.model.eval()
            self.loaded = True

    def available(self) -> bool:
        return self.loaded

    def predict(self, sequence: List[GazePoint]) -> Optional[Tuple[float, float, float]]:
        if not self.loaded or self.model is None:
            return None
        with torch.no_grad():
            tensor = torch.tensor([[p.x, p.y] for p in sequence], dtype=torch.float32).unsqueeze(0)
            output = self.model(tensor)
            pred_x, pred_y = output.squeeze(0).tolist()
            return float(pred_x), float(pred_y), 0.6

    def status(self) -> str:
        return "weights loaded" if self.loaded else "weights missing"
