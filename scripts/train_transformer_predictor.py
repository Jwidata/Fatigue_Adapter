import json
from pathlib import Path

import numpy as np
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:
    print(f"Torch not available: {exc}")
    raise SystemExit(1)


class TransformerGazeModel(nn.Module):
    def __init__(self, input_size: int = 5, d_model: int = 32, nhead: int = 2, num_layers: int = 1):
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


def main():
    dataset_path = Path("data/gaze_prediction_dataset.npz")
    if not dataset_path.exists():
        print("Dataset not found. Run scripts/build_gaze_prediction_dataset.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = np.load(dataset_path)
    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.float32)
    X_val = torch.tensor(data["X_val"], dtype=torch.float32)
    y_val = torch.tensor(data["y_val"], dtype=torch.float32)
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    if len(X_train) < 10:
        print("Warning: very small training set. Training will be noisy.")

    torch.manual_seed(7)
    input_size = X_train.shape[2]
    model = TransformerGazeModel(input_size=input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64,
        shuffle=True,
        pin_memory=device.type == "cuda",
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=64,
        shuffle=False,
        pin_memory=device.type == "cuda",
        num_workers=0,
    )

    history = {"train_loss": [], "val_loss": []}
    epochs = 40
    patience = 3
    best_val = None
    patience_left = patience
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x.to(device))
            loss = loss_fn(preds, batch_y.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_losses = []
            for batch_x, batch_y in val_loader:
                val_preds = model(batch_x.to(device))
                val_losses.append(loss_fn(val_preds, batch_y.to(device)).item())
            val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0

        history["train_loss"].append(epoch_loss / max(len(train_loader), 1))
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch+1}: train {history['train_loss'][-1]:.4f}, val {val_loss:.4f}")

        if best_val is None or val_loss < best_val:
            best_val = val_loss
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered")
                break

    weights_path = Path("configs/transformer_gaze_weights.pt")
    torch.save(model.state_dict(), weights_path)
    history_path = Path("artifacts/transformer_history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    config_path = Path("artifacts/transformer_config.json")
    config_path.write_text(json.dumps({"epochs": epoch + 1, "seed": 7, "input_size": input_size}, indent=2), encoding="utf-8")
    print(f"Saved weights to {weights_path}")
    print(f"Saved weights to {weights_path}")


if __name__ == "__main__":
    main()
