"""Converter stub for GazeBase to replay format."""

from pathlib import Path


def main():
    source = Path("Data/GazeBase_v2_0")
    target = Path("configs/replay_gaze.csv")
    print(f"TODO: Convert gaze data from {source} into {target}")


if __name__ == "__main__":
    main()
