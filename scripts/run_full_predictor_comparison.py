import subprocess


def run(cmd):
    print(f"\n>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    run("python scripts/build_gaze_prediction_dataset.py")
    run("python scripts/train_lstm_predictor.py")
    run("python scripts/train_gru_predictor.py")
    run("python scripts/train_cnn_predictor.py")
    run("python scripts/train_transformer_predictor.py")
    run("python scripts/train_xgboost_predictor.py")
    run("python scripts/evaluate_predictors.py")


if __name__ == "__main__":
    main()
