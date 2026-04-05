from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import (
    AdaptationOutcome,
    AdaptationResponse,
    CaseListResponse,
    GazeIngestRequest,
    ModeRequest,
    PolicyInfoResponse,
    PredictionEvalStats,
    PredictionInfoResponse,
    PredictionResponse,
    RiskResponse,
    RoiResponse,
    SliceImageResponse,
    StateResponse,
)


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
    return context.roi.get_rois(case_id, slice_id)


@router.post("/gaze")
def ingest_gaze(payload: GazeIngestRequest, request: Request):
    context = get_context(request)
    context.gaze.add_point(payload.point)
    return {"status": "ok"}


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
