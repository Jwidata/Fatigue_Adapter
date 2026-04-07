import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

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

try:
    from app.services.catalog_service import CatalogService
    from app.services.roi_service import RoiService
    from app.utils.config_utils import load_config
    ROI_AVAILABLE = True
except Exception:
    ROI_AVAILABLE = False


class LSTMGazeModel(nn.Module if nn is not None else object):
    def __init__(self, input_size: int = 5, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        output, _ = self.lstm(x)
        last = output[:, -1, :]
        return self.fc(last)


class GRUGazeModel(nn.Module if nn is not None else object):
    def __init__(self, input_size: int = 5, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        output, _ = self.gru(x)
        last = output[:, -1, :]
        return self.fc(last)


class TransformerGazeModel(nn.Module if nn is not None else object):
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


class TemporalCNN(nn.Module if nn is not None else object):
    def __init__(self, input_size: int = 5):
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
        "within_75": float((errors_px <= 75).mean() * 100),
        "within_100": float((errors_px <= 100).mean() * 100),
        "mean_error_norm": float(errors_norm.mean()),
        "median_error_norm": float(np.median(errors_norm)),
    }


def point_to_bbox_distance(px, py, x, y, w, h):
    if x <= px <= x + w and y <= py <= y + h:
        return 0.0
    dx = max(x - px, 0.0, px - (x + w))
    dy = max(y - py, 0.0, py - (y + h))
    return float(np.sqrt(dx * dx + dy * dy))


def point_in_polygon(px, py, points):
    inside = False
    if not points:
        return False
    j = len(points) - 1
    for i, (xi, yi) in enumerate(points):
        xj, yj = points[j]
        intersects = (yi > py) != (yj > py) and px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def roi_metrics(preds_norm, case_ids, slice_ids, width, height):
    if not ROI_AVAILABLE:
        print("ROI metrics unavailable: roi_import_failed")
        return {"roi_in": None, "roi_near": None, "roi_error": None}
    config = load_config()
    near_threshold = float(config.get("roi_near_threshold_px", 30))
    catalog = CatalogService(config).load_or_build()
    roi_service = RoiService(catalog, config)
    default_bbox = type(
        "DefaultBBox",
        (),
        {"x": 0.35, "y": 0.35, "w": 0.3, "h": 0.3},
    )()
    default_roi = type(
        "DefaultRoi",
        (),
        {"bbox": default_bbox, "polygon": None, "mask": None},
    )()

    inside = 0
    near = 0
    total = 0
    dist_sum = 0.0
    skip_reasons = {}
    for idx, case_id in enumerate(case_ids):
        if not case_id:
            skip_reasons["missing_case_id_default_roi"] = skip_reasons.get("missing_case_id_default_roi", 0) + 1
            rois = [default_roi]
            meta_width = width
            meta_height = height
        else:
            slice_id = int(slice_ids[idx])
            try:
                roi_resp = roi_service.get_rois(case_id, slice_id)
                rois = roi_resp.rois
                if not rois:
                    skip_reasons["no_rois"] = skip_reasons.get("no_rois", 0) + 1
                    continue
                try:
                    meta = catalog.get_slice_meta(case_id, slice_id)
                    meta_width = meta.width
                    meta_height = meta.height
                except Exception:
                    meta_width = width
                    meta_height = height
            except Exception:
                skip_reasons["roi_fetch_failed"] = skip_reasons.get("roi_fetch_failed", 0) + 1
                continue
        px = preds_norm[idx, 0] * meta_width
        py = preds_norm[idx, 1] * meta_height
        in_any = False
        near_any = False
        min_center_dist = None
        valid_shape = False
        for roi in rois:
            in_roi = False
            dist_to_edge = None
            center_dist = None
            if roi.bbox:
                x = roi.bbox.x * meta_width
                y = roi.bbox.y * meta_height
                w = roi.bbox.w * meta_width
                h = roi.bbox.h * meta_height
                dist_to_edge = point_to_bbox_distance(px, py, x, y, w, h)
                in_roi = dist_to_edge == 0.0
                cx = x + w / 2
                cy = y + h / 2
                center_dist = float(np.sqrt((px - cx) ** 2 + (py - cy) ** 2))
                valid_shape = True
            elif roi.polygon and roi.polygon.points:
                points = [(p[0] * meta_width, p[1] * meta_height) for p in roi.polygon.points]
                in_roi = point_in_polygon(px, py, points)
                center_x = sum(p[0] for p in points) / len(points)
                center_y = sum(p[1] for p in points) / len(points)
                center_dist = float(np.sqrt((px - center_x) ** 2 + (py - center_y) ** 2))
                dist_to_edge = center_dist
                valid_shape = True
            else:
                if roi.mask:
                    skip_reasons["roi_mask_unsupported"] = skip_reasons.get("roi_mask_unsupported", 0) + 1
                continue

            if center_dist is None:
                continue
            if min_center_dist is None or center_dist < min_center_dist:
                min_center_dist = center_dist
            if in_roi:
                in_any = True
            if dist_to_edge is not None and dist_to_edge <= near_threshold:
                near_any = True
            elif center_dist <= near_threshold:
                near_any = True

        if not valid_shape:
            skip_reasons["roi_shape_missing"] = skip_reasons.get("roi_shape_missing", 0) + 1
            continue
        if min_center_dist is None:
            skip_reasons["roi_center_missing"] = skip_reasons.get("roi_center_missing", 0) + 1
            continue
        total += 1
        dist_sum += min_center_dist
        if in_any:
            inside += 1
        if near_any:
            near += 1

    if total == 0:
        if skip_reasons:
            print(f"ROI metrics skipped for all samples: {skip_reasons}")
        return {"roi_in": None, "roi_near": None, "roi_error": None}
    if skip_reasons:
        print(f"ROI metrics skipped samples: {skip_reasons}")
    return {
        "roi_in": (inside / total) * 100,
        "roi_near": (near / total) * 100,
        "roi_error": dist_sum / total,
    }


def feature_indices(feature_names):
    if not feature_names:
        return {}
    return {name: idx for idx, name in enumerate(feature_names)}


def last_xy(seq, indices):
    idx_x = indices.get("x")
    idx_y = indices.get("y")
    if idx_x is None or idx_y is None:
        return None
    return float(seq[-1][idx_x]), float(seq[-1][idx_y])


def baseline_delta(seq, indices):
    idx_dx = indices.get("dx")
    idx_dy = indices.get("dy")
    if idx_dx is not None and idx_dy is not None:
        return float(seq[-1][idx_dx]), float(seq[-1][idx_dy])
    idx_x = indices.get("x")
    idx_y = indices.get("y")
    if idx_x is None or idx_y is None or len(seq) < 2:
        return None
    return float(seq[-1][idx_x] - seq[-2][idx_x]), float(seq[-1][idx_y] - seq[-2][idx_y])


def reconstruct_absolute(targets, sequences, indices, target_mode):
    preds = np.zeros((len(targets), 2), dtype=np.float32)
    for i, seq in enumerate(sequences):
        if target_mode == "absolute":
            preds[i] = targets[i]
            continue
        last_pos = last_xy(seq, indices)
        if last_pos is None:
            raise ValueError("missing_x_y_features")
        if target_mode == "delta":
            preds[i] = np.array(last_pos) + targets[i]
        elif target_mode == "residual":
            base = baseline_delta(seq, indices)
            if base is None:
                raise ValueError("missing_dx_dy_features")
            preds[i] = np.array(last_pos) + np.array(base) + targets[i]
        else:
            raise ValueError(f"unknown_target_mode:{target_mode}")
    return preds


def main():
    dataset_path = Path("data/gaze_prediction_dataset.npz")
    if not dataset_path.exists():
        print("Dataset not found. Run scripts/build_gaze_prediction_dataset.py first.")
        return

    summary_path = Path("data/gaze_prediction_dataset_summary.json")
    width = 512
    height = 512
    target_mode = "delta"
    feature_names = []
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        width = summary.get("width", 512)
        height = summary.get("height", 512)
        target_mode = summary.get("target", summary.get("target_mode", "delta"))
        feature_names = summary.get("feature_names", [])
        print(json.dumps(summary, indent=2))

    lstm_weights = Path("configs/lstm_gaze_weights.pt")
    transformer_weights = Path("configs/transformer_gaze_weights.pt")
    gru_weights = Path("configs/gru_gaze_weights.pt")
    cnn_weights = Path("configs/cnn_gaze_weights.pt")
    x_model_x = Path("configs/xgboost_gaze_model_x.json")
    x_model_y = Path("configs/xgboost_gaze_model_y.json")

    device = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
    if TORCH_AVAILABLE:
        print(f"Using device: {device}")

    data = np.load(dataset_path, allow_pickle=True)
    X_test = data["X_test"]
    y_target = data["y_test"]
    case_ids = data.get("case_ids", np.array([""] * len(X_test)))
    slice_ids = data.get("slice_ids", np.zeros(len(X_test)))

    if len(X_test) < 5:
        print("Not enough test samples to evaluate predictors.")
        return

    results = {}

    indices = feature_indices(feature_names)
    try:
        actual = reconstruct_absolute(y_target, X_test, indices, target_mode)
    except ValueError as exc:
        actual = None
        results["heuristic"] = {"status": "skipped", "reason": str(exc)}
        results["constant_velocity"] = results["heuristic"].copy()
    else:
        preds = []
        for seq in X_test:
            last_pos = last_xy(seq, indices)
            base = baseline_delta(seq, indices)
            if last_pos is None or base is None:
                preds.append((np.nan, np.nan))
                continue
            preds.append((last_pos[0] + base[0], last_pos[1] + base[1]))
        preds = np.array(preds, dtype=np.float32)
        valid = np.isfinite(preds).all(axis=1)
        if not valid.any():
            results["heuristic"] = {"status": "skipped", "reason": "missing_x_y_features"}
            results["constant_velocity"] = results["heuristic"].copy()
        else:
            preds = preds[valid]
            actual_valid = actual[valid]
            errors_norm = np.linalg.norm(preds - actual_valid, axis=1)
            errors_px = np.linalg.norm((preds - actual_valid) * np.array([width, height]), axis=1)
            metrics = compute_metrics(errors_px, errors_norm)
            metrics.update(roi_metrics(preds, case_ids[valid], slice_ids[valid], width, height))
            results["heuristic"] = {"status": "evaluated", "reason": "ok", **metrics}
            results["constant_velocity"] = results["heuristic"].copy()

    def eval_torch(name, model_cls, weights_path):
        if not TORCH_AVAILABLE:
            results[name] = {"status": "skipped", "reason": "torch_unavailable"}
            return
        if not weights_path.exists():
            results[name] = {"status": "skipped", "reason": "missing_weights"}
            return
        try:
            model = model_cls(input_size=X_test.shape[2])
            model.load_state_dict(torch.load(weights_path, map_location=device))
            model.to(device)
            model.eval()
            with torch.no_grad():
                pred_target = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
            preds_local = reconstruct_absolute(pred_target, X_test, indices, target_mode)
            actual_local = actual if actual is not None else reconstruct_absolute(y_target, X_test, indices, target_mode)
            errors_norm_local = np.linalg.norm(preds_local - actual_local, axis=1)
            errors_px_local = np.linalg.norm((preds_local - actual_local) * np.array([width, height]), axis=1)
            metrics_local = compute_metrics(errors_px_local, errors_norm_local)
            metrics_local.update(roi_metrics(preds_local, case_ids, slice_ids, width, height))
            results[name] = {"status": "evaluated", "reason": "ok", **metrics_local}
        except Exception as exc:
            results[name] = {"status": "failed", "reason": f"config_mismatch: {exc}"}

    eval_torch("lstm", LSTMGazeModel, lstm_weights)
    eval_torch("gru", GRUGazeModel, gru_weights)
    eval_torch("transformer", TransformerGazeModel, transformer_weights)
    eval_torch("cnn", TemporalCNN, cnn_weights)

    if not XGBOOST_AVAILABLE:
        results["xgboost"] = {"status": "skipped", "reason": "xgboost_unavailable"}
    elif x_model_x.exists() and x_model_y.exists():
        model_x = xgb.XGBRegressor()
        model_y = xgb.XGBRegressor()
        model_x.load_model(x_model_x)
        model_y.load_model(x_model_y)
        X_flat = X_test.reshape(X_test.shape[0], -1)
        pred_x = model_x.predict(X_flat)
        pred_y = model_y.predict(X_flat)
        pred_target = np.stack([pred_x, pred_y], axis=1)
        try:
            preds = reconstruct_absolute(pred_target, X_test, indices, target_mode)
            actual_local = actual if actual is not None else reconstruct_absolute(y_target, X_test, indices, target_mode)
            errors_norm = np.linalg.norm(preds - actual_local, axis=1)
            errors_px = np.linalg.norm((preds - actual_local) * np.array([width, height]), axis=1)
            metrics = compute_metrics(errors_px, errors_norm)
            metrics.update(roi_metrics(preds, case_ids, slice_ids, width, height))
            results["xgboost"] = {"status": "evaluated", "reason": "ok", **metrics}
        except Exception as exc:
            results["xgboost"] = {"status": "failed", "reason": f"config_mismatch: {exc}"}
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
                "within_75",
                "within_100",
                "mean_error_norm",
                "median_error_norm",
                "roi_in",
                "roi_near",
                "roi_error",
            ]
        )
        for name, metrics in results.items():
            if metrics.get("status") != "evaluated":
                writer.writerow([name, metrics.get("status"), metrics.get("reason"), "", "", "", "", "", "", "", "", "", "", "", ""])
            else:
                writer.writerow(
                    [
                        name,
                        metrics.get("status"),
                        metrics.get("reason"),
                        metrics.get("mean_error"),
                        metrics.get("median_error"),
                        metrics.get("rmse"),
                        metrics.get("within_25"),
                        metrics.get("within_50"),
                        metrics.get("within_75"),
                        metrics.get("within_100"),
                        metrics.get("mean_error_norm"),
                        metrics.get("median_error_norm"),
                        metrics.get("roi_in"),
                        metrics.get("roi_near"),
                        metrics.get("roi_error"),
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
        within_75 = [results[name]["within_75"] for name in plot_names]
        within_100 = [results[name]["within_100"] for name in plot_names]
        x = np.arange(len(plot_names))
        plt.bar(x - 0.3, within_25, width=0.2, label="Within 25 px")
        plt.bar(x - 0.1, within_50, width=0.2, label="Within 50 px")
        plt.bar(x + 0.1, within_75, width=0.2, label="Within 75 px")
        plt.bar(x + 0.3, within_100, width=0.2, label="Within 100 px")
        plt.xticks(x, plot_names)
        plt.title("Prediction accuracy within thresholds")
        plt.ylabel("Percent")
        plt.legend()
        plt.savefig(plots_dir / "threshold_accuracy_comparison.png")

        roi_vals = [results[name].get("roi_in") for name in plot_names]
        if any(v is not None for v in roi_vals):
            plt.figure()
            plt.bar(plot_names, [v or 0 for v in roi_vals])
            plt.title("Predictions inside ROI")
            plt.ylabel("Percent")
            plt.savefig(plots_dir / "roi_in_comparison.png")

        roi_near_vals = [results[name].get("roi_near") for name in plot_names]
        if any(v is not None for v in roi_near_vals):
            plt.figure()
            plt.bar(plot_names, [v or 0 for v in roi_near_vals])
            plt.title("Predictions near ROI")
            plt.ylabel("Percent")
            plt.savefig(plots_dir / "roi_near_comparison.png")

        roi_error_vals = [results[name].get("roi_error") for name in plot_names]
        if any(v is not None for v in roi_error_vals):
            plt.figure()
            plt.bar(plot_names, [v or 0 for v in roi_error_vals])
            plt.title("ROI distance error")
            plt.ylabel("Pixels")
            plt.savefig(plots_dir / "roi_error_comparison.png")

    for name, metrics in results.items():
        if metrics.get("status") == "evaluated":
            print(f"{name}: evaluated")
        else:
            print(f"{name}: skipped because {metrics.get('reason')}")

    evaluated = {k: v for k, v in results.items() if v.get("status") == "evaluated"}
    if evaluated:
        best_mean = min(evaluated.items(), key=lambda x: x[1]["mean_error"])
        best_median = min(evaluated.items(), key=lambda x: x[1]["median_error"])
        best_within25 = max(evaluated.items(), key=lambda x: x[1]["within_25"])
        best_within50 = max(evaluated.items(), key=lambda x: x[1]["within_50"])
        best_within75 = max(evaluated.items(), key=lambda x: x[1]["within_75"])
        best_within100 = max(evaluated.items(), key=lambda x: x[1]["within_100"])
        roi_candidates = {k: v for k, v in evaluated.items() if v.get("roi_in") is not None}
        summary = {
            "best_by_mean_error": best_mean[0],
            "best_by_median_error": best_median[0],
            "best_by_within_25": best_within25[0],
            "best_by_within_50": best_within50[0],
            "best_by_within_75": best_within75[0],
            "best_by_within_100": best_within100[0],
        }
        if roi_candidates:
            summary["best_by_roi_in"] = max(roi_candidates.items(), key=lambda x: x[1]["roi_in"])[0]
            roi_error_candidates = {k: v for k, v in roi_candidates.items() if v.get("roi_error") is not None}
            if roi_error_candidates:
                summary["best_by_roi_error"] = min(roi_error_candidates.items(), key=lambda x: x[1]["roi_error"])[0]
        summary_path = Path("artifacts/eval/predictor_ranked_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
