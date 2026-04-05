# Fatigue_Adapter

Local-first feasibility prototype for predictive gaze-aware adaptation during medical image viewing. It loads CT slices and ROI, predicts the next gaze location, estimates attention-risk, and applies proactive UI support. Built to swap simulated gaze for Tobii and swap demo image sources later without touching core services.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` in a browser.

The demo reads from `Data/medicalimages` and will auto-index DICOM series into `data/catalog.json`.
If you add new data or want to refresh SEG mappings, delete `data/catalog.json` and restart the app.

## Project layout

- `app/main.py` FastAPI entrypoint and static UI
- `app/api/routes.py` REST endpoints
- `app/services/` core services (catalog, ROI, gaze, attention, state, adaptation)
- `app/adapters/` integration points for gaze and image adapters
- `app/utils/` DICOM and geometry utilities
- `app/static/` HTML/CSS/JS UI
- `configs/` JSON config and manual ROI examples
- `scripts/` helper scripts

## Architecture

The app follows a lightweight clean architecture:

- **Catalog + Image services**: scan `Data/medicalimages`, detect DICOM series, load a slice, render to PNG.
- **ROI service**: resolves important regions from DICOM-SEG when available, or manual ROI JSON fallback.
- **Gaze adapters**: synthetic, replay, and mouse are unified under one schema.
- **Predictor framework**: heuristic baseline plus optional learned predictors when weights exist.
- **Additional baselines**: GRU, temporal CNN, XGBoost, constant-velocity.
- **Attention-risk service**: checks predicted gaze against ROI.
- **Policy service**: selects proactive UI actions based on risk.
- **Outcome + learning stubs**: log whether adaptation helped and record transitions for RL-ready experimentation.

## Important regions (ROI)

The ROI service (`app/services/roi_service.py`) normalizes everything to a common schema:

- BBoxes in normalized coordinates (0-1)
- Optional polygons or masks (mask placeholder for now)

SEG is parsed when available; manual ROI fallback still applies via `configs/manual_rois.json`.

## Gaze coordinate mapping

The browser maps screen coordinates to image coordinates using the image element bounds:

- Screen → image pixel space in `app/static/app.js`
- Gaze is stored in image pixels on the backend
- ROI checks convert normalized ROI → image pixels

Pan/zoom can be added later by expanding the mapping logic on the frontend and the same ROI utilities on the backend.

## Swapping in Tobii

- Add a new adapter in `app/adapters/gaze/tobii.py`
- Implement the `GazeAdapter` interface (`next_point`)
- Update `GazeManager` to register the adapter and accept `gaze_source="tobii"`

The rest of the pipeline stays unchanged because all gaze points share a common schema.

## Swapping in real medical pipelines

- Replace `CatalogService` and `ImageService` to pull from PACS, streaming, or other sources
- Add a `RealDicomSegAdapter` implementation in `app/services/roi_service.py`

The UI and state logic remain intact because they consume a stable ROI and image interface.

## Predictive loop

The current demo follows this flow:

`recent gaze -> predicted next look -> risk -> action -> outcome`

Policy modes:

- **Quiet**: show only high-risk interventions
- **Balanced**: show medium/high risk when ROI priority is high
- **Aggressive**: adapt early and more often

Calmness controls (configurable):

- minimum risk to display a message
- minimum risk to trigger intervention
- cooldown between interventions
- message hold duration
- silent low-risk mode

## Prediction transparency

The UI includes a "Prediction Transparency" panel that reports:

- predictor type (heuristic, LSTM, Transformer)
- whether weights are loaded
- confidence and evaluation status

If no accuracy is measured yet, the UI explicitly states this.

Calm messaging is intentional: the UI only surfaces suggestions when risk rises or an intervention is triggered.

## Prediction evaluation hooks

The prototype logs prediction error by comparing the last predicted point to the next observed gaze point.
Debug mode shows rolling mean/median error and percentage within 25/50 px.
Raw evaluation logs are stored in `data/prediction_eval.jsonl`.

## Dataset inspection + subsets

Run `python scripts/inspect_full_imaging_dataset.py` to generate:

- `data/full_dataset_report.json`
- `data/full_dataset_report.csv`
- `data/relevant_cases_ranked.csv`
- `data/demo_subset_cases.txt`
- `data/eval_subset_cases.txt`

ROI sanity checks are saved to `artifacts/roi_checks/`.

## Dataset modes

Configure in `configs/default_config.json`:

- `dataset.mode`: `full`, `demo_subset`, or `eval_subset`
- `dataset.case_list_file`: optional manual case list

## Limitations

- Only one slice is used for state computation (default slice 0)
- DICOM-SEG parsing is minimal (bbox from masks)
- Prediction is heuristic by default (LSTM scaffold is optional)
- Replay gaze expects a simple CSV format in `configs/replay_gaze.csv` (GazeBase conversion stub)
- No persistence or multi-user state

## API endpoints

- `GET /api/cases`
- `GET /api/cases/{case_id}/slices/{slice_id}`
- `GET /api/roi/{case_id}/{slice_id}`
- `POST /api/gaze`
- `GET /api/state`
- `GET /api/prediction`
- `GET /api/prediction/info`
- `GET /api/prediction/metrics`
- `GET /api/risk`
- `GET /api/adaptation`
- `GET /api/adaptation/outcome`
- `GET /api/policy/info`
- `GET /api/predictors`
- `POST /api/mode`
- `POST /api/reset`

## Scripts

`scripts/preload_case.py` builds a catalog index in `data/catalog.json`.

## Predictor modes

Configured in `configs/default_config.json`:

- `predictor_mode: heuristic` uses velocity-based prediction
- `predictor_mode: lstm` uses the LSTM scaffold (requires weights)
- `predictor_mode: transformer` uses the Transformer scaffold (requires weights)
- `predictor_mode: auto` uses Transformer if available, else LSTM, else heuristic

If LSTM weights are missing, the system logs a fallback and continues.
If Transformer weights are missing, the system falls back the same way.

## RL-ready logging

Adaptation outcomes are logged for later policy learning in `data/policy_transitions.jsonl`.
Each transition records `state`, `action`, `reward`, and `next_state`.

## Predictor framework

Predictors:

- **Heuristic**: next gaze estimated from recent motion direction/velocity
- **Constant velocity**: interpretable motion baseline
- **LSTM**: learned recurrent sequence model (requires weights)
- **Transformer**: learned attention-based sequence model (requires weights)
- **GRU**: lightweight recurrent baseline (requires weights)
- **Temporal CNN**: 1D convolution baseline (requires weights)
- **XGBoost**: engineered-feature baseline (requires weights)

The system never claims a learned model is active unless weights are loaded.
If weights are missing, it falls back to heuristic and reports the fallback in the UI.

## Training + evaluation scripts

- Build dataset: `python scripts/build_gaze_prediction_dataset.py`
- Train LSTM: `python scripts/train_lstm_predictor.py`
- Train GRU: `python scripts/train_gru_predictor.py`
- Train CNN: `python scripts/train_cnn_predictor.py`
- Train Transformer: `python scripts/train_transformer_predictor.py`
- Train XGBoost: `python scripts/train_xgboost_predictor.py`
- Evaluate: `python scripts/evaluate_predictors.py`
- Full pipeline: `python scripts/run_full_predictor_comparison.py`
- Full dataset selection + eval: `python scripts/run_dataset_selection_and_model_eval.py`

Evaluation outputs:

- `artifacts/eval/predictor_results.json`
- `artifacts/eval/predictor_results.csv`
- `artifacts/eval/predictor_ranked_summary.json`
- plots in `artifacts/plots/`

## Plots generated

- Mean prediction error (bar)
- Median prediction error (bar)
- RMSE prediction error (bar)
- % within 25 px and 50 px (bar)
- GRU training loss curve (if available)
- CNN training loss curve (if available)
- LSTM training loss curve (if available)
- Transformer training loss curve (if available)

## Dataset features

The prediction dataset normalizes gaze by image width/height and includes derived features:

- x_norm, y_norm
- dx, dy
- speed
- direction angle
