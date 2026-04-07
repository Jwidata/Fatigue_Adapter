import argparse
import json
import time
from typing import Dict
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description="Stream gaze samples to the API")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/api/gaze_stream")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--case-id", type=str, default="")
    parser.add_argument("--slice-id", type=int, default=0)
    parser.add_argument("--source", type=str, default="synthetic")
    parser.add_argument("--mode", type=str, default="normal")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    interval = 1.0 / max(args.hz, 1.0)
    steps = int(args.duration * args.hz)
    start = time.time()

    for i in range(steps):
        t = time.time()
        phase = (i / max(steps, 1)) * 6.283185307179586
        x = (args.width / 2) + (args.width * 0.2) * (0.5 * (1 + __import__("math").sin(phase)))
        y = (args.height / 2) + (args.height * 0.2) * (0.5 * (1 + __import__("math").cos(phase)))
        payload: Dict[str, object] = {
            "timestamp": t * 1000,
            "x": float(x),
            "y": float(y),
            "source": args.source,
            "mode": args.mode,
        }
        if args.case_id:
            payload["case_id"] = args.case_id
            payload["slice_id"] = args.slice_id
        req = Request(
            args.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req) as resp:
                body = resp.read().decode("utf-8")
                if not args.quiet and (args.print_every > 0) and (i % args.print_every == 0):
                    print(resp.status, body)
        except Exception as exc:
            if not args.quiet:
                print(f"error: {exc}")
        elapsed = time.time() - t
        sleep_for = interval - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

    print(f"Sent {steps} samples in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
