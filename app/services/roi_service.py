from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pydicom

from app.models.schemas import RoiBBox, RoiResponse, RoiShape
from app.services.catalog_service import CatalogService
from app.utils.geometry_utils import normalize_bbox


class RealDicomSegAdapter:
    """Placeholder adapter for future DICOM SEG parsing."""

    def __init__(self, catalog: CatalogService, config: Dict):
        self.catalog = catalog
        self.config = config

    def get_rois(self, case_id: str, slice_id: int) -> Optional[List[RoiShape]]:
        seg_paths = self.catalog.get_seg_paths(case_id)
        if not seg_paths:
            self._log("seg_not_found", case_id, slice_id, extra="no_seg_paths")
            return None

        for seg_path in seg_paths:
            rois = self._parse_seg_file(case_id, seg_path, slice_id)
            if rois:
                self._log("seg_ok", case_id, slice_id, extra=f"rois={len(rois)}")
                return rois
        self._log("seg_no_match", case_id, slice_id, extra="no_frames_mapped")
        return None

    def _parse_seg_file(self, case_id: str, seg_path: str, slice_id: int) -> Optional[List[RoiShape]]:
        try:
            ds = pydicom.dcmread(seg_path)
        except Exception as exc:
            self._log("seg_read_fail", "", slice_id, extra=str(exc))
            return None

        if getattr(ds, "Modality", "") != "SEG":
            return None

        sop_uid_map = self._resolve_sop_map(case_id, ds)
        if not sop_uid_map:
            self._log("seg_not_found", case_id, slice_id, extra="no_sop_uid_map")
            return None

        try:
            pixel_array = ds.pixel_array
        except Exception as exc:
            self._log("seg_pixel_fail", "", slice_id, extra=str(exc))
            return None

        if pixel_array.ndim == 2:
            frames = [pixel_array]
        else:
            frames = list(pixel_array)

        slice_masks: Dict[int, np.ndarray] = {}
        for frame_index, frame in enumerate(frames):
            sop_uid = self._extract_frame_sop_uid(ds, frame_index)
            if not sop_uid or sop_uid not in sop_uid_map:
                continue
            mapped_slice = sop_uid_map[sop_uid]
            if mapped_slice not in slice_masks:
                slice_masks[mapped_slice] = frame > 0
            else:
                slice_masks[mapped_slice] = np.logical_or(slice_masks[mapped_slice], frame > 0)

        if slice_id not in slice_masks:
            return None

        mask = slice_masks[slice_id]
        if not np.any(mask):
            return None

        ys, xs = np.where(mask)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        height, width = mask.shape
        nx, ny, nw, nh = normalize_bbox((x_min, y_min, x_max - x_min + 1, y_max - y_min + 1), width, height)

        return [
            RoiShape(
                id=f"seg-{slice_id}",
                label="SEG ROI",
                type="bbox",
                priority=0.9,
                bbox=RoiBBox(x=nx, y=ny, w=nw, h=nh),
            )
        ]

    def _resolve_sop_map(self, case_id: str, ds) -> Dict[str, int]:
        series_uid = self._extract_referenced_series_uid(ds)
        if series_uid:
            sop_map = self.catalog.get_sop_uid_map_for_series_uid(case_id, series_uid)
            if sop_map:
                return sop_map
        return self.catalog.get_sop_uid_map(case_id)

    @staticmethod
    def _extract_referenced_series_uid(ds) -> Optional[str]:
        if hasattr(ds, "ReferencedSeriesSequence"):
            try:
                ref = ds.ReferencedSeriesSequence[0]
                uid = getattr(ref, "SeriesInstanceUID", None)
                if uid:
                    return str(uid)
            except Exception:
                return None
        return None

    @staticmethod
    def _extract_frame_sop_uid(ds, frame_index: int) -> Optional[str]:
        if hasattr(ds, "PerFrameFunctionalGroupsSequence"):
            try:
                frame = ds.PerFrameFunctionalGroupsSequence[frame_index]
                derivation = getattr(frame, "DerivationImageSequence", None)
                if derivation:
                    for item in derivation:
                        source_seq = getattr(item, "SourceImageSequence", None)
                        if not source_seq:
                            continue
                        for source in source_seq:
                            uid = getattr(source, "ReferencedSOPInstanceUID", None)
                            if uid:
                                return str(uid)
            except Exception:
                return None
        return None

    def _log(self, event: str, case_id: str, slice_id: int, extra: str = ""):
        if not self.config.get("roi_debug"):
            return
        message = f"SEG {event} case={case_id} slice={slice_id}"
        if extra:
            message += f" {extra}"
        print(message)


class RoiService:
    def __init__(self, catalog: CatalogService, config: Dict):
        self.catalog = catalog
        self.config = config
        self._manual_path = Path(__file__).resolve().parents[2] / "configs" / "manual_rois.json"
        self._override_path = Path(__file__).resolve().parents[2] / "data" / "roi_overrides.json"
        self._manual = self._load_manual()
        self._overrides = self._load_overrides()
        self._seg_adapter = RealDicomSegAdapter(catalog, config)

    def _load_manual(self) -> Dict:
        if not self._manual_path.exists():
            return {}
        with self._manual_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_overrides(self) -> Dict:
        if not self._override_path.exists():
            return {}
        with self._override_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_overrides(self) -> None:
        self._override_path.parent.mkdir(parents=True, exist_ok=True)
        with self._override_path.open("w", encoding="utf-8") as handle:
            json.dump(self._overrides, handle, indent=2)

    def set_override(self, case_id: str, slice_id: int, roi: RoiShape) -> None:
        self._overrides.setdefault("cases", {}).setdefault(case_id, {}).setdefault("slices", {})[
            str(slice_id)
        ] = {"rois": [roi.model_dump()]}
        self._save_overrides()

    def clear_override(self, case_id: str, slice_id: int) -> None:
        case_blob = self._overrides.get("cases", {}).get(case_id, {})
        slice_blob = case_blob.get("slices", {})
        if str(slice_id) in slice_blob:
            slice_blob.pop(str(slice_id), None)
            self._save_overrides()

    def get_rois(self, case_id: str, slice_id: int) -> RoiResponse:
        meta = self.catalog.get_slice_meta(case_id, slice_id)
        rois = self._override_rois(case_id, slice_id)
        source = "user"
        if rois is None:
            rois = self._seg_adapter.get_rois(case_id, slice_id)
            source = "seg"
        if rois is None:
            rois = self._manual_rois(case_id, slice_id)
            source = "manual"
        if not rois:
            rois = [
                RoiShape(
                    id="roi-default",
                    label="Default ROI",
                    type="bbox",
                    priority=0.5,
                    bbox=RoiBBox(x=0.35, y=0.35, w=0.3, h=0.3),
                )
            ]
            source = "none"
        if self.config.get("roi_debug"):
            print(
                f"ROI source={source} case={case_id} slice={slice_id} count={len(rois)} "
                f"image={meta.width}x{meta.height}"
            )
        return RoiResponse(
            case_id=case_id,
            slice_id=slice_id,
            rois=rois,
            source=source,
            image_width=meta.width,
            image_height=meta.height,
        )

    def _override_rois(self, case_id: str, slice_id: int) -> Optional[List[RoiShape]]:
        case_blob = self._overrides.get("cases", {}).get(case_id, {})
        slice_blob = case_blob.get("slices", {}).get(str(slice_id), {})
        rois: List[RoiShape] = []
        for entry in slice_blob.get("rois", []):
            rois.append(self._build_roi(entry))
        return rois if rois else None

    def _manual_rois(self, case_id: str, slice_id: int) -> List[RoiShape]:
        rois: List[RoiShape] = []
        case_blob = self._manual.get("cases", {}).get(case_id, {})
        slice_blob = case_blob.get("slices", {}).get(str(slice_id), {})
        for entry in slice_blob.get("rois", []):
            rois.append(self._build_roi(entry))

        if not rois:
            default_blob = self._manual.get("default", {})
            for entry in default_blob.get("rois", []):
                rois.append(self._build_roi(entry))
        return rois

    @staticmethod
    def _build_roi(entry: Dict) -> RoiShape:
        roi_type = entry.get("type", "bbox")
        bbox = entry.get("bbox")
        return RoiShape(
            id=entry.get("id", "roi"),
            label=entry.get("label", "ROI"),
            type=roi_type,
            priority=float(entry.get("priority", 0.5)),
            bbox=RoiBBox(**bbox) if bbox else None,
            polygon=entry.get("polygon"),
            mask=entry.get("mask"),
        )
