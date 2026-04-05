from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from app.models.schemas import GazePoint
from app.services.predictors.base_predictor import BasePredictor


if nn is not None:
    class TransformerGazeModel(nn.Module):
        def __init__(self, input_size: int = 2, d_model: int = 32, nhead: int = 2, num_layers: int = 1):
            super().__init__()
            self.input = nn.Linear(input_size, d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output = nn.Linear(d_model, 2)

        def forward(self, x):
            embedded = self.input(x)
            encoded = self.encoder(embedded)
            last = encoded[:, -1, :]
            return self.output(last)
else:
    TransformerGazeModel = None


class TransformerPredictor(BasePredictor):
    name = "transformer"

    def __init__(self, weights_path: Path):
        self.weights_path = weights_path
        self.loaded = False
        self.model = None
        if torch is None or nn is None or TransformerGazeModel is None:
            return
        if weights_path.exists():
            self.model = TransformerGazeModel()
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
