import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.adapters.gaze.synthetic import SyntheticGazeAdapter


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic gaze log")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--mode", type=str, default="normal")
    parser.add_argument("--case-id", type=str, default="")
    parser.add_argument("--slice-id", type=int, default=0)
    args = parser.parse_args()

    adapter = SyntheticGazeAdapter(mode=args.mode)
    log_path = Path("data/gaze_log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = 0.0
    with log_path.open("w", encoding="utf-8") as handle:
        for _ in range(args.count):
            point = adapter.next_point(args.width, args.height, timestamp)
            payload = {
                "timestamp": timestamp,
                "x": point.x,
                "y": point.y,
                "source": "synthetic",
                "mode": args.mode,
                "case_id": args.case_id or None,
                "slice_id": args.slice_id,
            }
            handle.write(json.dumps(payload) + "\n")
            timestamp += args.interval_ms

    print(f"Wrote {args.count} points to {log_path}")


if __name__ == "__main__":
    main()
