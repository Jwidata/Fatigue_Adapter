from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from app.services.feature_builder import FeatureSequence
from app.services.predictors.base_predictor import BasePredictor
from app.services.predictors.predictor_utils import reconstruct_absolute


if nn is not None:
    class LSTMGazeModel(nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden_size, 2)

        def forward(self, x):
            output, _ = self.lstm(x)
            last = output[:, -1, :]
            return self.fc(last)
else:
    LSTMGazeModel = None


class LSTMPredictor(BasePredictor):
    name = "lstm"

    def __init__(self, weights_path: Path, input_size: int, target_mode: str):
        self.weights_path = weights_path
        self.input_size = input_size
        self.target_mode = target_mode
        self.loaded = False
        self.model = None
        self.device = None
        self.load_error = ""
        if torch is None or nn is None or LSTMGazeModel is None:
            return
        if weights_path.exists():
            try:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model = LSTMGazeModel(input_size=input_size)
                state = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state)
                self.model.to(self.device)
                self.model.eval()
                self.loaded = True
            except Exception as exc:
                self.load_error = str(exc)
                self.model = None
                self.loaded = False

    def available(self) -> bool:
        return self.loaded

    def predict(self, sequence: FeatureSequence) -> Optional[Tuple[float, float, float]]:
        if not self.loaded or self.model is None:
            return None
        with torch.no_grad():
            tensor = torch.tensor(sequence.features, dtype=torch.float32).unsqueeze(0)
            if self.device is not None:
                tensor = tensor.to(self.device)
            output = self.model(tensor)
            pred_x, pred_y = output.squeeze(0).tolist()
            norm_x, norm_y = reconstruct_absolute(
                pred_x,
                pred_y,
                sequence.last_pos_norm,
                sequence.last_delta_norm,
                sequence.target_mode,
            )
            return float(norm_x), float(norm_y), 0.6

    def status(self) -> str:
        if self.loaded:
            return "weights loaded"
        if self.load_error:
            return f"load_failed: {self.load_error}"
        return "weights missing"
