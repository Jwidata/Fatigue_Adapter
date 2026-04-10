from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi.responses import Response
from pydantic import BaseModel, Field


class SeriesInfo(BaseModel):
    case_id: str
    series_id: str
    slice_count: int


class CaseInfo(BaseModel):
    case_id: str
    series: List[SeriesInfo]


class CaseListResponse(BaseModel):
    cases: List[CaseInfo]


class SliceMeta(BaseModel):
    case_id: str
    slice_id: int
    width: int
    height: int
    window: Optional[List[float]] = None


@dataclass
class SliceImageResponse:
    bytes: bytes
    meta: SliceMeta

    def to_response(self) -> Response:
        headers = {
            "X-Case-ID": self.meta.case_id,
            "X-Slice-ID": str(self.meta.slice_id),
            "X-Image-Width": str(self.meta.width),
            "X-Image-Height": str(self.meta.height),
        }
        if self.meta.window:
            headers["X-Window"] = ",".join(str(v) for v in self.meta.window)
        return Response(content=self.bytes, media_type="image/png", headers=headers)


class RoiBBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class RoiPolygon(BaseModel):
    points: List[List[float]]


class RoiMask(BaseModel):
    encoded: str = Field(..., description="Placeholder for future mask encoding")


class RoiShape(BaseModel):
    id: str
    label: str
    type: str
    priority: float = 0.5
    bbox: Optional[RoiBBox] = None
    polygon: Optional[RoiPolygon] = None
    mask: Optional[RoiMask] = None


class RoiResponse(BaseModel):
    case_id: str
    slice_id: int
    rois: List[RoiShape]
    source: str
    image_width: int
    image_height: int


class GazePoint(BaseModel):
    timestamp: float
    x: float
    y: float
    source: str
    mode: Optional[str] = None
    pupil: Optional[float] = None
    blink: Optional[bool] = None


class GazeIngestRequest(BaseModel):
    point: GazePoint


class GazeStreamRequest(BaseModel):
    timestamp: float
    x: float
    y: float
    case_id: Optional[str] = None
    slice_id: Optional[int] = None
    source: Optional[str] = None
    mode: Optional[str] = None


class DisplayGazeRequest(BaseModel):
    timestamp: float
    x: float
    y: float
    screen_width: int
    screen_height: int
    case_id: Optional[str] = None
    slice_id: Optional[int] = None
    source: Optional[str] = None
    mode: Optional[str] = None
    normalized: Optional[bool] = None


class ModeRequest(BaseModel):
    gaze_mode: str
    gaze_source: str
    case_id: Optional[str] = None
    slice_id: Optional[int] = None
    predictive_enabled: Optional[bool] = None
    adaptive_enabled: Optional[bool] = None
    policy_mode: Optional[str] = None
    message_min_risk: Optional[str] = None
    intervention_min_risk: Optional[str] = None
    cooldown_ms: Optional[int] = None
    message_hold_ms: Optional[int] = None
    silent_low_risk: Optional[bool] = None


class RoiOverrideRequest(BaseModel):
    case_id: str
    slice_id: int
    bbox: RoiBBox
    label: Optional[str] = None
    priority: Optional[float] = None


class ViewportRequest(BaseModel):
    case_id: str
    slice_id: int
    image_left: float
    image_top: float
    image_width: float
    image_height: float
    image_pixel_width: int
    image_pixel_height: int
    screen_width: int
    screen_height: int
    timestamp: float


class MetricsResponse(BaseModel):
    dwell_time_ms: float
    roi_hits: int
    first_fixation_delay_ms: Optional[float]
    roi_coverage_pct: float
    dispersion: float
    scan_coverage_pct: float


class PredictionPoint(BaseModel):
    x: float
    y: float
    confidence: float


class PredictionResponse(BaseModel):
    predicted: Optional[PredictionPoint]
    method: str
    timestamp: float
    based_on: int


class RoiStatusResponse(BaseModel):
    inside_roi: bool
    distance_px: Optional[float]
    roi_center_x: Optional[float] = None
    roi_center_y: Optional[float] = None
    roi_bbox: Optional[RoiBBox] = None
    reason: Optional[str] = None


class PredictionEvalStats(BaseModel):
    count: int
    mean_error_px: Optional[float]
    median_error_px: Optional[float]
    within_25_px_pct: Optional[float]
    within_50_px_pct: Optional[float]


class PredictionInfoResponse(BaseModel):
    predictor_mode: str
    active_predictor: str
    explanation: str
    model_status: str
    confidence: Optional[float]
    accuracy_note: str
    eval_stats: Optional[PredictionEvalStats] = None


class RiskResponse(BaseModel):
    risk_level: str
    reason: str
    predicted_in_roi: bool
    roi_priority: float


class PolicyInfoResponse(BaseModel):
    policy_mode: str
    message_min_risk: str
    intervention_min_risk: str
    cooldown_ms: int
    message_hold_ms: int
    last_intervention_ts: Optional[float] = None
    silent_low_risk: bool = False


class StateResponse(BaseModel):
    state: str
    metrics: MetricsResponse
    latest_gaze: Optional[GazePoint]
    window_ms: int
    prediction: Optional[PredictionResponse] = None
    risk: Optional[RiskResponse] = None
    attention_status: Optional[str] = None


class RealtimePredictionResponse(BaseModel):
    gaze: Optional[GazePoint]
    prediction: Optional[PredictionPoint]
    model_used: str
    roi_status: Optional[RoiStatusResponse]
    adaptation: Optional[AdaptationResponse]
    risk: Optional[RiskResponse]
    buffer_len: int
    sequence_len: int


class AdaptationCommand(BaseModel):
    state: str
    actions: List[str]
    highlight_strength: float
    dim_level: float
    message: Optional[str] = None


class AdaptationResponse(BaseModel):
    command: AdaptationCommand
    timestamp: float
    metrics: Optional[MetricsResponse] = None


class AdaptationOutcome(BaseModel):
    adaptation_id: str
    action: str
    success: Optional[bool]
    time_to_roi_ms: Optional[float]
    notes: Optional[str] = None
    timestamp: float
