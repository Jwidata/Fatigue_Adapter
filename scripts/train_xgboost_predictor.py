import json
from pathlib import Path

import numpy as np

try:
    import xgboost as xgb
except Exception as exc:
    print(f"XGBoost not available: {exc}")
    raise SystemExit(1)


def main():
    dataset_path = Path("data/gaze_prediction_dataset.npz")
    if not dataset_path.exists():
        print("Dataset not found. Run scripts/build_gaze_prediction_dataset.py first.")
        return

    data = np.load(dataset_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    if len(X_train_flat) < 10:
        print("Warning: very small training set. Training will be noisy.")

    params = {"max_depth": 4, "n_estimators": 200, "learning_rate": 0.05, "subsample": 0.8}
    model_x = xgb.XGBRegressor(**params)
    model_y = xgb.XGBRegressor(**params)

    model_x.fit(X_train_flat, y_train[:, 0])
    model_y.fit(X_train_flat, y_train[:, 1])

    weights_path = Path("configs/xgboost_gaze_model.json")
    model_x.get_booster().save_model(weights_path.with_name("xgboost_gaze_model_x.json"))
    model_y.get_booster().save_model(weights_path.with_name("xgboost_gaze_model_y.json"))

    history_path = Path("artifacts/xgboost_config.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps({"params": params}, indent=2), encoding="utf-8")
    print("Saved XGBoost models to configs/xgboost_gaze_model_x.json and ..._y.json")


if __name__ == "__main__":
    main()
