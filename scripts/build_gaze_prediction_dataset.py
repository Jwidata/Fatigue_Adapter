import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

try:
    from app.services.catalog_service import CatalogService
    from app.services.roi_service import RoiService
    from app.utils.config_utils import load_config
    ROI_AVAILABLE = True
except Exception:
    ROI_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(description="Build gaze prediction dataset")
    parser.add_argument("--case-list", type=str, default="")
    parser.add_argument("--sequence-len", type=int, default=30)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--features", type=str, default="x,y,dx,dy,speed,accel,dir_sin,dir_cos,dist_roi,inside_roi,dist_roi_edge")
    parser.add_argument("--target-mode", type=str, default="delta")
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()

    log_path = Path("data/gaze_log.jsonl")
    if not log_path.exists():
        print("No gaze log found at data/gaze_log.jsonl")
        return

    sequence_len = args.sequence_len
    sequences = []
    targets = []

    points = []
    case_ids = []
    slice_ids = []
    case_filter = None
    if args.case_list:
        case_filter = {line.strip() for line in Path(args.case_list).read_text(encoding="utf-8").splitlines() if line.strip()}
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            case_id = item.get("case_id")
            if case_filter and case_id not in case_filter:
                continue
            points.append(
                (
                    item["timestamp"],
                    item["x"],
                    item["y"],
                    item.get("case_id") or "",
                    int(item.get("slice_id") or 0),
                )
            )

    width = float(args.width)
    height = float(args.height)
    smooth_window = max(args.smooth_window, 1)
    horizon = max(args.horizon, 1)
    feature_set = {f.strip() for f in args.features.split(",") if f.strip()}
    feature_order = [
        "x",
        "y",
        "dx",
        "dy",
        "speed",
        "accel",
        "dir_sin",
        "dir_cos",
        "dist_roi",
        "inside_roi",
        "dist_roi_edge",
    ]
    feature_names = [name for name in feature_order if name in feature_set]

    roi_service = None
    catalog = None
    if ROI_AVAILABLE:
        config = load_config()
        catalog = CatalogService(config).load_or_build()
        roi_service = RoiService(catalog, config)

    for idx in range(len(points) - sequence_len - horizon):
        seq = points[idx : idx + sequence_len]
        target = points[idx + sequence_len + horizon - 1]
        feature_seq = []
        xs = [p[1] for p in seq]
        ys = [p[2] for p in seq]
        if smooth_window > 1:
            xs = [sum(xs[max(0, i - smooth_window + 1) : i + 1]) / (i - max(0, i - smooth_window + 1) + 1) for i in range(len(xs))]
            ys = [sum(ys[max(0, i - smooth_window + 1) : i + 1]) / (i - max(0, i - smooth_window + 1) + 1) for i in range(len(ys))]

        roi_center = None
        roi_bbox = None
        case_id = seq[-1][3]
        slice_id = seq[-1][4]
        if roi_service and case_id:
            try:
                roi_resp = roi_service.get_rois(case_id, int(slice_id))
                if roi_resp.rois:
                    roi = roi_resp.rois[0]
                    if roi.bbox:
                        roi_bbox = roi.bbox
                        cx = roi_bbox.x + roi_bbox.w / 2
                        cy = roi_bbox.y + roi_bbox.h / 2
                        roi_center = (cx, cy)
            except Exception:
                roi_center = None

        for j, (ts, _, _, _, _) in enumerate(seq):
            x = xs[j]
            y = ys[j]
            x_norm = x / width
            y_norm = y / height
            if j == 0:
                dx = 0.0
                dy = 0.0
            else:
                prev_x = xs[j - 1]
                prev_y = ys[j - 1]
                dx = (x - prev_x) / width
                dy = (y - prev_y) / height
            speed = math.sqrt(dx * dx + dy * dy)
            if j >= 2:
                prev2_x = xs[j - 2]
                prev2_y = ys[j - 2]
                prev_dx = (prev_x - prev2_x) / width
                prev_dy = (prev_y - prev2_y) / height
                accel = math.sqrt((dx - prev_dx) ** 2 + (dy - prev_dy) ** 2)
            else:
                accel = 0.0
            angle = math.atan2(dy, dx) if speed > 0 else 0.0
            dir_sin = math.sin(angle)
            dir_cos = math.cos(angle)
            dist_roi = 0.0
            inside_roi = 0.0
            dist_roi_edge = 0.0
            if roi_center is not None and roi_bbox is not None:
                dist_roi = math.sqrt((x_norm - roi_center[0]) ** 2 + (y_norm - roi_center[1]) ** 2)
                inside_roi = 1.0 if (roi_bbox.x <= x_norm <= roi_bbox.x + roi_bbox.w and roi_bbox.y <= y_norm <= roi_bbox.y + roi_bbox.h) else 0.0
                left = abs(x_norm - roi_bbox.x)
                right = abs(x_norm - (roi_bbox.x + roi_bbox.w))
                top = abs(y_norm - roi_bbox.y)
                bottom = abs(y_norm - (roi_bbox.y + roi_bbox.h))
                dist_roi_edge = min(left, right, top, bottom)

            features = []
            if "x" in feature_set:
                features.append(x_norm)
            if "y" in feature_set:
                features.append(y_norm)
            if "dx" in feature_set:
                features.append(dx)
            if "dy" in feature_set:
                features.append(dy)
            if "speed" in feature_set:
                features.append(speed)
            if "accel" in feature_set:
                features.append(accel)
            if "dir_sin" in feature_set:
                features.append(dir_sin)
            if "dir_cos" in feature_set:
                features.append(dir_cos)
            if "dist_roi" in feature_set:
                features.append(dist_roi)
            if "inside_roi" in feature_set:
                features.append(inside_roi)
            if "dist_roi_edge" in feature_set:
                features.append(dist_roi_edge)
            feature_seq.append(features)
        sequences.append(feature_seq)
        prev = seq[-1]
        if args.target_mode == "absolute":
            targets.append([target[1] / width, target[2] / height])
        elif args.target_mode == "residual":
            baseline_dx = (prev[1] - seq[-2][1]) / width
            baseline_dy = (prev[2] - seq[-2][2]) / height
            target_dx = (target[1] - prev[1]) / width
            target_dy = (target[2] - prev[2]) / height
            targets.append([target_dx - baseline_dx, target_dy - baseline_dy])
        else:
            target_dx = (target[1] - prev[1]) / width
            target_dy = (target[2] - prev[2]) / height
            targets.append([target_dx, target_dy])
        case_ids.append(target[3])
        slice_ids.append(target[4])

    if not sequences:
        print("Not enough gaze points to build dataset")
        return

    sequences = np.array(sequences, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    case_ids = np.array(case_ids)
    slice_ids = np.array(slice_ids)

    total = len(sequences)
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)

    dataset_path = Path("data/gaze_prediction_dataset.npz")
    np.savez(
        dataset_path,
        X_train=sequences[:train_end],
        y_train=targets[:train_end],
        X_val=sequences[train_end:val_end],
        y_val=targets[train_end:val_end],
        X_test=sequences[val_end:],
        y_test=targets[val_end:],
        case_ids=case_ids[val_end:],
        slice_ids=slice_ids[val_end:],
    )
    metadata = {
        "sequence_len": sequence_len,
        "num_train_samples": int(train_end),
        "num_val_samples": int(val_end - train_end),
        "num_test_samples": int(total - val_end),
        "feature_shape": list(sequences.shape[1:]),
        "target_shape": list(targets.shape[1:]),
        "feature_dim": sequences.shape[2] if sequences.ndim == 3 else 0,
         "feature_names": feature_names,
        "normalized": True,
        "width": args.width,
        "height": args.height,
        "target": args.target_mode,
        "smooth_window": smooth_window,
        "horizon": horizon,
    }
    metadata_path = Path("data/gaze_prediction_dataset_summary.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved dataset to {dataset_path}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
