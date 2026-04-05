import subprocess
from pathlib import Path


def run(cmd):
    print(f"\n>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    run("python scripts/inspect_full_imaging_dataset.py")
    if Path("data/demo_subset_cases.txt").exists():
        run("python scripts/roi_sanity_check.py")

    case_list = "data/eval_subset_cases.txt" if Path("data/eval_subset_cases.txt").exists() else ""
    dataset_cmd = "python scripts/build_gaze_prediction_dataset.py"
    if case_list:
        dataset_cmd += f" --case-list {case_list}"
    run(dataset_cmd)

    run("python scripts/train_lstm_predictor.py")
    run("python scripts/train_gru_predictor.py")
    run("python scripts/train_cnn_predictor.py")
    run("python scripts/train_transformer_predictor.py")
    run("python scripts/train_xgboost_predictor.py")
    run("python scripts/evaluate_predictors.py")


if __name__ == "__main__":
    main()
