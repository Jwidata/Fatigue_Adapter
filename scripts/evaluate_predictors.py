import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    xgb = None
    XGBOOST_AVAILABLE = False


class LSTMGazeModel(nn.Module if nn is not None else object):
    def __init__(self, input_size: int = 6, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        output, _ = self.lstm(x)
        last = output[:, -1, :]
        return self.fc(last)


class GRUGazeModel(nn.Module if nn is not None else object):
    def __init__(self, input_size: int = 6, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        output, _ = self.gru(x)
        last = output[:, -1, :]
        return self.fc(last)


class TransformerGazeModel(nn.Module if nn is not None else object):
    def __init__(self, input_size: int = 6, d_model: int = 32, nhead: int = 2, num_layers: int = 1):
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


class TemporalCNN(nn.Module if nn is not None else object):
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


def compute_metrics(errors_px, errors_norm):
    errors_px = np.array(errors_px)
    errors_norm = np.array(errors_norm)
    return {
        "mean_error": float(errors_px.mean()),
        "median_error": float(np.median(errors_px)),
        "rmse": float(np.sqrt(np.mean(errors_px**2))),
        "within_25": float((errors_px <= 25).mean() * 100),
        "within_50": float((errors_px <= 50).mean() * 100),
        "mean_error_norm": float(errors_norm.mean()),
        "median_error_norm": float(np.median(errors_norm)),
    }


def main():
    dataset_path = Path("data/gaze_prediction_dataset.npz")
    if not dataset_path.exists():
        print("Dataset not found. Run scripts/build_gaze_prediction_dataset.py first.")
        return

    summary_path = Path("data/gaze_prediction_dataset_summary.json")
    width = 512
    height = 512
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        width = summary.get("width", 512)
        height = summary.get("height", 512)
        print(json.dumps(summary, indent=2))

    lstm_weights = Path("configs/lstm_gaze_weights.pt")
    transformer_weights = Path("configs/transformer_gaze_weights.pt")
    gru_weights = Path("configs/gru_gaze_weights.pt")
    cnn_weights = Path("configs/cnn_gaze_weights.pt")
    x_model_x = Path("configs/xgboost_gaze_model_x.json")
    x_model_y = Path("configs/xgboost_gaze_model_y.json")

    print(f"LSTM weights path: {lstm_weights} (exists={lstm_weights.exists()})")
    print(f"Transformer weights path: {transformer_weights} (exists={transformer_weights.exists()})")
    print(f"GRU weights path: {gru_weights} (exists={gru_weights.exists()})")
    print(f"CNN weights path: {cnn_weights} (exists={cnn_weights.exists()})")
    print(f"XGBoost weights path: {x_model_x} (exists={x_model_x.exists()})")
    registry = {
        "heuristic": "available",
        "constant_velocity": "available",
        "lstm": "available" if lstm_weights.exists() else "missing_weights",
        "gru": "available" if gru_weights.exists() else "missing_weights",
        "temporal_cnn": "available" if cnn_weights.exists() else "missing_weights",
        "transformer": "available" if transformer_weights.exists() else "missing_weights",
        "xgboost": "available" if x_model_x.exists() and x_model_y.exists() else "missing_weights",
    }
    print("Predictor registry summary:")
    for name, status in registry.items():
        print(f"- {name}: {status}")

    device = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
    if TORCH_AVAILABLE:
        print(f"Using device: {device}")

    data = np.load(dataset_path)
    X_test = data["X_test"]
    y_test = data["y_test"]
    X_xy = X_test
    y_xy = y_test

    if len(X_test) < 5:
        print("Not enough test samples to evaluate predictors.")
        return

    results = {}

    # Heuristic baseline (constant velocity from last step)
    preds = []
    for seq in X_test:
        x_curr, y_curr, dx, dy = seq[-1][0], seq[-1][1], seq[-1][2], seq[-1][3]
        pred_x = x_curr + dx
        pred_y = y_curr + dy
        preds.append((pred_x, pred_y))
    preds = np.array(preds)
    errors_norm = np.linalg.norm(preds - y_xy, axis=1)
    errors_px = np.linalg.norm((preds - y_xy) * np.array([width, height]), axis=1)
    results["heuristic"] = {"status": "evaluated", "reason": "ok", **compute_metrics(errors_px, errors_norm)}

    # Constant-velocity baseline (same as heuristic for now, labeled separately)
    results["constant_velocity"] = results["heuristic"].copy()

    if not TORCH_AVAILABLE:
        results["lstm"] = {"status": "skipped", "reason": "torch_unavailable"}
    elif lstm_weights.exists():
        try:
            model = LSTMGazeModel(input_size=X_xy.shape[2])
            model.load_state_dict(torch.load(lstm_weights, map_location=device))
            model.to(device)
            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(X_xy, dtype=torch.float32).to(device)).cpu().numpy()
            errors_norm = np.linalg.norm(preds - y_xy, axis=1)
            errors_px = np.linalg.norm((preds - y_xy) * np.array([width, height]), axis=1)
            results["lstm"] = {"status": "evaluated", "reason": "ok", **compute_metrics(errors_px, errors_norm)}
        except Exception as exc:
            results["lstm"] = {"status": "failed", "reason": f"config_mismatch: {exc}"}
    else:
        results["lstm"] = {"status": "skipped", "reason": "missing_weights"}

    if not TORCH_AVAILABLE:
        results["gru"] = {"status": "skipped", "reason": "torch_unavailable"}
    elif gru_weights.exists():
        model = GRUGazeModel(input_size=X_xy.shape[2])
        model.load_state_dict(torch.load(gru_weights, map_location=device))
        model.to(device)
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_xy, dtype=torch.float32).to(device)).cpu().numpy()
        errors_norm = np.linalg.norm(preds - y_xy, axis=1)
        errors_px = np.linalg.norm((preds - y_xy) * np.array([width, height]), axis=1)
        results["gru"] = {"status": "evaluated", "reason": "ok", **compute_metrics(errors_px, errors_norm)}
    else:
        results["gru"] = {"status": "skipped", "reason": "missing_weights"}

    if not TORCH_AVAILABLE:
        results["transformer"] = {"status": "skipped", "reason": "torch_unavailable"}
    elif transformer_weights.exists():
        try:
            model = TransformerGazeModel(input_size=X_xy.shape[2])
            model.load_state_dict(torch.load(transformer_weights, map_location=device))
            model.to(device)
            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(X_xy, dtype=torch.float32).to(device)).cpu().numpy()
            errors_norm = np.linalg.norm(preds - y_xy, axis=1)
            errors_px = np.linalg.norm((preds - y_xy) * np.array([width, height]), axis=1)
            results["transformer"] = {"status": "evaluated", "reason": "ok", **compute_metrics(errors_px, errors_norm)}
        except Exception as exc:
            results["transformer"] = {"status": "failed", "reason": f"config_mismatch: {exc}"}
    else:
        results["transformer"] = {"status": "skipped", "reason": "missing_weights"}

    if not TORCH_AVAILABLE:
        results["cnn"] = {"status": "skipped", "reason": "torch_unavailable"}
    elif cnn_weights.exists():
        model = TemporalCNN(input_size=X_xy.shape[2])
        model.load_state_dict(torch.load(cnn_weights, map_location=device))
        model.to(device)
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_xy, dtype=torch.float32).to(device)).cpu().numpy()
        errors_norm = np.linalg.norm(preds - y_xy, axis=1)
        errors_px = np.linalg.norm((preds - y_xy) * np.array([width, height]), axis=1)
        results["cnn"] = {"status": "evaluated", "reason": "ok", **compute_metrics(errors_px, errors_norm)}
    else:
        results["cnn"] = {"status": "skipped", "reason": "missing_weights"}

    if not XGBOOST_AVAILABLE:
        results["xgboost"] = {"status": "skipped", "reason": "xgboost_unavailable"}
    elif x_model_x.exists() and x_model_y.exists():
        model_x = xgb.XGBRegressor()
        model_y = xgb.XGBRegressor()
        model_x.load_model(x_model_x)
        model_y.load_model(x_model_y)
        X_flat = X_xy.reshape(X_xy.shape[0], -1)
        pred_x = model_x.predict(X_flat)
        pred_y = model_y.predict(X_flat)
        preds = np.stack([pred_x, pred_y], axis=1)
        errors_norm = np.linalg.norm(preds - y_xy, axis=1)
        errors_px = np.linalg.norm((preds - y_xy) * np.array([width, height]), axis=1)
        results["xgboost"] = {"status": "evaluated", "reason": "ok", **compute_metrics(errors_px, errors_norm)}
    else:
        results["xgboost"] = {"status": "skipped", "reason": "missing_weights"}

    summary_path = Path("artifacts/eval/predictor_results.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    csv_path = Path("artifacts/eval/predictor_results.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "predictor",
                "status",
                "reason",
                "mean_error",
                "median_error",
                "rmse",
                "within_25",
                "within_50",
                "mean_error_norm",
                "median_error_norm",
            ]
        )
        for name, metrics in results.items():
            if metrics.get("status") != "evaluated":
                writer.writerow([name, metrics.get("status"), metrics.get("reason"), "", "", "", "", "", "", ""])
            else:
                writer.writerow(
                    [
                        name,
                        metrics.get("status"),
                        metrics.get("reason"),
                        metrics["mean_error"],
                        metrics["median_error"],
                        metrics["rmse"],
                        metrics["within_25"],
                        metrics["within_50"],
                        metrics["mean_error_norm"],
                        metrics["median_error_norm"],
                    ]
                )

    plots_dir = Path("artifacts/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_names = [name for name, metrics in results.items() if metrics.get("status") == "evaluated"]
    if plot_names:
        plt.figure()
        plt.bar(plot_names, [results[name]["mean_error"] for name in plot_names])
        plt.title("Mean prediction error")
        plt.ylabel("Pixels")
        plt.savefig(plots_dir / "mean_error_comparison.png")

        plt.figure()
        plt.bar(plot_names, [results[name]["median_error"] for name in plot_names])
        plt.title("Median prediction error")
        plt.ylabel("Pixels")
        plt.savefig(plots_dir / "median_error_comparison.png")

        plt.figure()
        plt.bar(plot_names, [results[name]["rmse"] for name in plot_names])
        plt.title("RMSE prediction error")
        plt.ylabel("Pixels")
        plt.savefig(plots_dir / "rmse_comparison.png")

        plt.figure()
        within_25 = [results[name]["within_25"] for name in plot_names]
        within_50 = [results[name]["within_50"] for name in plot_names]
        x = np.arange(len(plot_names))
        plt.bar(x - 0.2, within_25, width=0.4, label="Within 25 px")
        plt.bar(x + 0.2, within_50, width=0.4, label="Within 50 px")
        plt.xticks(x, plot_names)
        plt.title("Prediction accuracy within thresholds")
        plt.ylabel("Percent")
        plt.legend()
        plt.savefig(plots_dir / "threshold_accuracy_comparison.png")

    gru_history = Path("artifacts/gru_history.json")
    if gru_history.exists():
        history = json.loads(gru_history.read_text(encoding="utf-8"))
        plt.figure()
        plt.plot(history.get("train_loss", []), label="train")
        plt.plot(history.get("val_loss", []), label="val")
        plt.title("GRU training loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.savefig(plots_dir / "gru_training_loss.png")

    cnn_history = Path("artifacts/cnn_history.json")
    if cnn_history.exists():
        history = json.loads(cnn_history.read_text(encoding="utf-8"))
        plt.figure()
        plt.plot(history.get("train_loss", []), label="train")
        plt.plot(history.get("val_loss", []), label="val")
        plt.title("Temporal CNN training loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.savefig(plots_dir / "cnn_training_loss.png")

    for name, metrics in results.items():
        if metrics.get("status") == "evaluated":
            print(f"{name}: evaluated")
        else:
            print(f"{name}: skipped because {metrics.get('reason')}")

    evaluated = {k: v for k, v in results.items() if v.get("status") == "evaluated"}
    if evaluated:
        best_mean = min(evaluated.items(), key=lambda x: x[1]["mean_error"])
        best_median = min(evaluated.items(), key=lambda x: x[1]["median_error"])
        best_within50 = max(evaluated.items(), key=lambda x: x[1]["within_50"])
        summary = {
            "best_by_mean_error": best_mean[0],
            "best_by_median_error": best_median[0],
            "best_by_within_50": best_within50[0],
        }
        summary_path = Path("artifacts/eval/predictor_ranked_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
