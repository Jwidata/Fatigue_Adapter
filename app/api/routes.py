from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import (
    AdaptationOutcome,
    AdaptationResponse,
    CaseListResponse,
    GazePoint,
    GazeIngestRequest,
    GazeStreamRequest,
    ModeRequest,
    PolicyInfoResponse,
    PredictionEvalStats,
    PredictionInfoResponse,
    PredictionResponse,
    RealtimePredictionResponse,
    RiskResponse,
    RoiResponse,
    RoiStatusResponse,
    SliceImageResponse,
    StateResponse,
)
from app.services.roi_status_service import compute_roi_status


router = APIRouter()


def get_context(request: Request):
    return request.app.state.context


@router.get("/cases", response_model=CaseListResponse)
def list_cases(request: Request):
    context = get_context(request)
    return CaseListResponse(cases=context.catalog.list_cases())


@router.get("/cases/{case_id}/slices/{slice_id}")
def get_slice_image(case_id: str, slice_id: int, request: Request):
    context = get_context(request)
    try:
        image_bytes, meta = context.images.get_slice_png(case_id, slice_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return SliceImageResponse(bytes=image_bytes, meta=meta).to_response()


@router.get("/roi/{case_id}/{slice_id}", response_model=RoiResponse)
def get_roi(case_id: str, slice_id: int, request: Request):
    context = get_context(request)
    roi_resp = context.roi.get_rois(case_id, slice_id)
    roi_log = []
    for roi in roi_resp.rois:
        mask_shape = None
        if roi.mask and hasattr(roi.mask, "encoded"):
            mask_shape = "encoded"
        roi_log.append({"bbox": roi.bbox.dict() if roi.bbox else None, "mask": mask_shape})
    print(
        {
            "roi_bbox": roi_log,
            "roi_mask_shape": [entry["mask"] for entry in roi_log],
            "image_size": [roi_resp.image_width, roi_resp.image_height],
        }
    )
    return roi_resp


@router.post("/gaze")
def ingest_gaze(payload: GazeIngestRequest, request: Request):
    context = get_context(request)
    context.gaze.add_point(payload.point)
    return {"status": "ok"}


@router.post("/gaze_stream", response_model=RealtimePredictionResponse)
def ingest_gaze_stream(payload: GazeStreamRequest, request: Request):
    context = get_context(request)
    if payload.case_id:
        context.state_service.set_active_case(payload.case_id, payload.slice_id)
        context.gaze.set_context(payload.case_id, payload.slice_id or 0)
    if payload.source:
        context.gaze.set_source(payload.source)
    if payload.mode:
        context.gaze.set_mode(payload.mode)
    point = GazePoint(
        timestamp=payload.timestamp,
        x=payload.x,
        y=payload.y,
        source=payload.source or context.gaze.source,
        mode=payload.mode or context.gaze.mode,
    )
    context.gaze.add_point(point)
    state = context.state_service.compute_state(sample_if_needed=False)
    prediction = state.prediction.predicted if state.prediction else None
    model_used = state.prediction.method if state.prediction else "none"
    roi_status = None
    meta = context.catalog.get_slice_meta(context.state_service.case_id, context.state_service.slice_id)
    if prediction:
        roi_status = compute_roi_status(prediction.x, prediction.y, context.roi.get_rois(context.state_service.case_id, context.state_service.slice_id).rois, meta.width, meta.height)
    return RealtimePredictionResponse(
        gaze=point,
        prediction=prediction,
        model_used=model_used,
        roi_status=roi_status,
        adaptation=context.policy.get_latest(),
        risk=state.risk,
        buffer_len=len(context.gaze.get_window_points()),
        sequence_len=context.prediction.sequence_len,
    )


@router.get("/state", response_model=StateResponse)
def get_state(request: Request):
    context = get_context(request)
    return context.state_service.compute_state()


@router.get("/prediction", response_model=PredictionResponse)
def get_prediction(request: Request):
    context = get_context(request)
    state = context.state_service.compute_state()
    if state.prediction is None:
        return PredictionResponse(predicted=None, method="disabled", timestamp=0.0, based_on=0)
    return state.prediction


@router.get("/predict", response_model=RealtimePredictionResponse)
def predict_realtime(request: Request):
    context = get_context(request)
    state = context.state_service.compute_state(sample_if_needed=False)
    prediction = state.prediction.predicted if state.prediction else None
    model_used = state.prediction.method if state.prediction else "none"
    latest = context.gaze.get_latest()
    roi_status = None
    meta = context.catalog.get_slice_meta(context.state_service.case_id, context.state_service.slice_id)
    if prediction:
        roi_status = compute_roi_status(prediction.x, prediction.y, context.roi.get_rois(context.state_service.case_id, context.state_service.slice_id).rois, meta.width, meta.height)
    return RealtimePredictionResponse(
        gaze=latest,
        prediction=prediction,
        model_used=model_used,
        roi_status=roi_status,
        adaptation=context.policy.get_latest(),
        risk=state.risk,
        buffer_len=len(context.gaze.get_window_points()),
        sequence_len=context.prediction.sequence_len,
    )


@router.get("/roi_status", response_model=RoiStatusResponse)
def get_roi_status(request: Request):
    context = get_context(request)
    latest = context.gaze.get_latest()
    if latest is None:
        return RoiStatusResponse(inside_roi=False, distance_px=None, reason="no_gaze")
    meta = context.catalog.get_slice_meta(context.state_service.case_id, context.state_service.slice_id)
    rois = context.roi.get_rois(context.state_service.case_id, context.state_service.slice_id).rois
    return compute_roi_status(latest.x, latest.y, rois, meta.width, meta.height)


@router.get("/prediction/info", response_model=PredictionInfoResponse)
def get_prediction_info(request: Request):
    context = get_context(request)
    return context.prediction.get_info()


@router.get("/prediction/metrics", response_model=PredictionEvalStats)
def get_prediction_metrics(request: Request):
    context = get_context(request)
    return context.prediction.get_eval_stats()


@router.get("/predictors")
def get_predictors(request: Request):
    context = get_context(request)
    return {
        "available": context.prediction.manager.available_predictors(),
        "active": context.prediction.manager.active_name,
        "status": context.prediction.manager.status(),
        "fallback_reason": context.prediction.manager.fallback_reason,
    }


@router.get("/predictor/results")
def get_predictor_results():
    from pathlib import Path
    import json

    path = Path("artifacts/eval/predictor_results.json")
    if not path.exists():
        return {"status": "missing", "results": {}}
    return {"status": "ok", "results": json.loads(path.read_text(encoding="utf-8"))}


@router.get("/dataset/summary")
def get_dataset_summary():
    from pathlib import Path
    import json

    path = Path("data/gaze_prediction_dataset_summary.json")
    if not path.exists():
        return {"status": "missing", "summary": {}}
    return {"status": "ok", "summary": json.loads(path.read_text(encoding="utf-8"))}


@router.get("/risk", response_model=RiskResponse)
def get_risk(request: Request):
    context = get_context(request)
    state = context.state_service.compute_state()
    if state.risk is None:
        return RiskResponse(risk_level="low", reason="no_state", predicted_in_roi=False, roi_priority=0.0)
    return state.risk


@router.get("/adaptation", response_model=AdaptationResponse)
def get_adaptation(request: Request):
    context = get_context(request)
    latest = context.policy.get_latest()
    if latest is None:
        context.state_service.compute_state()
        latest = context.policy.get_latest()
    return latest


@router.get("/policy/info", response_model=PolicyInfoResponse)
def get_policy_info(request: Request):
    context = get_context(request)
    return context.policy.get_policy_info()


@router.get("/adaptation/outcome", response_model=AdaptationOutcome)
def get_adaptation_outcome(request: Request):
    context = get_context(request)
    outcome = context.outcome.get_latest()
    if outcome is None:
        return AdaptationOutcome(
            adaptation_id="",
            action="",
            success=None,
            time_to_roi_ms=None,
            notes=None,
            timestamp=0.0,
        )
    return outcome


@router.post("/mode")
def set_mode(payload: ModeRequest, request: Request):
    context = get_context(request)
    context.gaze.set_mode(payload.gaze_mode)
    context.gaze.set_source(payload.gaze_source)
    if payload.predictive_enabled is not None:
        context.state_service.set_predictive_enabled(payload.predictive_enabled)
    if payload.adaptive_enabled is not None:
        context.policy.set_adaptive_enabled(payload.adaptive_enabled)
    if payload.policy_mode:
        context.policy.set_policy_mode(payload.policy_mode)
    context.policy.update_policy_config(
        message_min_risk=payload.message_min_risk,
        intervention_min_risk=payload.intervention_min_risk,
        cooldown_ms=payload.cooldown_ms,
        message_hold_ms=payload.message_hold_ms,
        silent_low_risk=payload.silent_low_risk,
    )
    if payload.case_id:
        context.state_service.set_active_case(payload.case_id, payload.slice_id)
    return {"status": "ok"}


@router.post("/reset")
def reset_session(request: Request):
    context = get_context(request)
    context.gaze.reset()
    context.state_service.reset()
    return {"status": "ok"}
