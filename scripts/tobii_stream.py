import argparse
import json
import time
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description="Stream Tobii gaze to the API")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/api/gaze_display")
    parser.add_argument("--screen-width", type=int, default=1920)
    parser.add_argument("--screen-height", type=int, default=1080)
    parser.add_argument("--case-id", type=str, default="")
    parser.add_argument("--slice-id", type=int, default=0)
    parser.add_argument("--source", type=str, default="tobii")
    parser.add_argument("--mode", type=str, default="normal")
    args = parser.parse_args()

    try:
        import tobii_research as tr
    except Exception as exc:
        raise SystemExit(f"tobii_research not available: {exc}")

    trackers = tr.find_all_eyetrackers()
    if not trackers:
        raise SystemExit("No Tobii eye trackers found")

    tracker = trackers[0]

    def gaze_callback(gaze_data):
        left = gaze_data.get("left_gaze_point_on_display_area")
        right = gaze_data.get("right_gaze_point_on_display_area")
        gaze = left if left and left[0] is not None else right
        if not gaze or gaze[0] is None or gaze[1] is None:
            return
        payload = {
            "timestamp": time.time() * 1000,
            "x": float(gaze[0]),
            "y": float(gaze[1]),
            "screen_width": args.screen_width,
            "screen_height": args.screen_height,
            "source": args.source,
            "mode": args.mode,
            "normalized": True,
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
                resp.read()
        except Exception:
            pass

    tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_callback, as_dictionary=True)
    print("Streaming Tobii gaze. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_callback)


if __name__ == "__main__":
    main()
