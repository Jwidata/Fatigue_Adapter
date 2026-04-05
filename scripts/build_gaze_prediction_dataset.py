import argparse
import json
import math
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Build gaze prediction dataset")
    parser.add_argument("--case-list", type=str, default="")
    parser.add_argument("--sequence-len", type=int, default=20)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()

    log_path = Path("data/gaze_log.jsonl")
    if not log_path.exists():
        print("No gaze log found at data/gaze_log.jsonl")
        return

    sequence_len = args.sequence_len
    sequences = []
    targets = []

    points = []
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
            points.append((item["timestamp"], item["x"], item["y"]))

    width = float(args.width)
    height = float(args.height)
    for idx in range(len(points) - sequence_len - 1):
        seq = points[idx : idx + sequence_len]
        target = points[idx + sequence_len]
        feature_seq = []
        for j, (ts, x, y) in enumerate(seq):
            x_norm = x / width
            y_norm = y / height
            if j == 0:
                dx = 0.0
                dy = 0.0
            else:
                prev = seq[j - 1]
                dx = (x - prev[1]) / width
                dy = (y - prev[2]) / height
            speed = math.sqrt(dx * dx + dy * dy)
            angle = math.atan2(dy, dx) if speed > 0 else 0.0
            feature_seq.append([x_norm, y_norm, dx, dy, speed, angle])
        sequences.append(feature_seq)
        targets.append([target[1] / width, target[2] / height])

    if not sequences:
        print("Not enough gaze points to build dataset")
        return

    sequences = np.array(sequences, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

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
    )
    metadata = {
        "sequence_len": sequence_len,
        "num_train_samples": int(train_end),
        "num_val_samples": int(val_end - train_end),
        "num_test_samples": int(total - val_end),
        "feature_shape": list(sequences.shape[1:]),
        "target_shape": list(targets.shape[1:]),
        "feature_dim": sequences.shape[2] if sequences.ndim == 3 else 0,
        "feature_names": ["x_norm", "y_norm", "dx", "dy", "speed", "angle"],
        "normalized": True,
        "width": args.width,
        "height": args.height,
    }
    metadata_path = Path("data/gaze_prediction_dataset_summary.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved dataset to {dataset_path}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
