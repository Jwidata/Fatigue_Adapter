from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pydicom

from app.models.schemas import CaseInfo, SeriesInfo, SliceMeta


class CatalogService:
    def __init__(self, config: Dict):
        self.config = config
        self._catalog: Dict[str, Dict] = {}
        self._data_root = Path(__file__).resolve().parents[2] / "Data" / "medicalimages"
        self._catalog_path = Path(__file__).resolve().parents[2] / "Data" / "catalog.json"
        self._catalog_path_fallback = Path(__file__).resolve().parents[2] / "data" / "catalog.json"

    def load_or_build(self):
        if self._catalog_path.exists():
            with self._catalog_path.open("r", encoding="utf-8") as handle:
                self._catalog = json.load(handle)
        elif self._catalog_path_fallback.exists():
            with self._catalog_path_fallback.open("r", encoding="utf-8") as handle:
                self._catalog = json.load(handle)
        else:
            self._catalog = self._build_catalog()
            self._catalog_path.parent.mkdir(parents=True, exist_ok=True)
            with self._catalog_path.open("w", encoding="utf-8") as handle:
                json.dump(self._catalog, handle, indent=2)
        self._apply_dataset_mode()
        return self

    def _apply_dataset_mode(self):
        dataset_cfg = self.config.get("dataset", {})
        mode = dataset_cfg.get("mode", "full")
        case_list_file = dataset_cfg.get("case_list_file")

        if mode == "full" and not case_list_file:
            return

        if mode == "demo_subset":
            case_list_path = Path(__file__).resolve().parents[2] / "Data" / "demo_subset_cases.txt"
            if not case_list_path.exists():
                case_list_path = Path(__file__).resolve().parents[2] / "data" / "demo_subset_cases.txt"
        elif mode == "eval_subset":
            case_list_path = Path(__file__).resolve().parents[2] / "Data" / "eval_subset_cases.txt"
            if not case_list_path.exists():
                case_list_path = Path(__file__).resolve().parents[2] / "data" / "eval_subset_cases.txt"
        else:
            case_list_path = Path(case_list_file) if case_list_file else None

        if not case_list_path or not case_list_path.exists():
            return

        case_ids = {line.strip() for line in case_list_path.read_text(encoding="utf-8").splitlines() if line.strip()}
        if not case_ids:
            return
        self._catalog = {case_id: data for case_id, data in self._catalog.items() if case_id in case_ids}

    def list_cases(self) -> List[CaseInfo]:
        cases: List[CaseInfo] = []
        for case_id, case_data in self._catalog.items():
            series = [
                SeriesInfo(
                    case_id=case_id,
                    series_id=series_id,
                    slice_count=series_data["slice_count"],
                )
                for series_id, series_data in case_data["series"].items()
            ]
            cases.append(CaseInfo(case_id=case_id, series=series))
        return cases

    def get_default_case(self) -> Tuple[str, str]:
        for case_id, case_data in self._catalog.items():
            for series_id in case_data["series"].keys():
                return case_id, series_id
        raise FileNotFoundError("No DICOM cases found in Data/medicalimages")

    def get_series(self, case_id: str, series_id: str) -> Dict:
        return self._catalog[case_id]["series"][series_id]

    def get_seg_paths(self, case_id: str) -> List[str]:
        return self._catalog.get(case_id, {}).get("segmentations", [])

    def get_sop_uid_map(self, case_id: str) -> Dict[str, int]:
        series_id = self._catalog[case_id]["default_series"]
        series = self.get_series(case_id, series_id)
        sop_uids = series.get("sop_instance_uids", [])
        return {uid: idx for idx, uid in enumerate(sop_uids) if uid}

    def get_sop_uid_map_for_series_uid(self, case_id: str, series_instance_uid: str) -> Dict[str, int]:
        series_map = self._catalog.get(case_id, {}).get("series", {})
        for series_data in series_map.values():
            if series_data.get("series_instance_uid") == series_instance_uid:
                sop_uids = series_data.get("sop_instance_uids", [])
                return {uid: idx for idx, uid in enumerate(sop_uids) if uid}
        return {}

    def get_slice_path(self, case_id: str, slice_id: int) -> str:
        series_id = self._catalog[case_id]["default_series"]
        series = self.get_series(case_id, series_id)
        slice_paths = series["slice_paths"]
        if slice_id < 0 or slice_id >= len(slice_paths):
            raise FileNotFoundError("Slice not found")
        return slice_paths[slice_id]

    def get_slice_meta(self, case_id: str, slice_id: int) -> SliceMeta:
        series_id = self._catalog[case_id]["default_series"]
        series = self.get_series(case_id, series_id)
        width = series.get("width", 0)
        height = series.get("height", 0)
        return SliceMeta(case_id=case_id, slice_id=slice_id, width=width, height=height)

    def _build_catalog(self) -> Dict:
        catalog: Dict[str, Dict] = {}
        for dirpath, _, filenames in os.walk(self._data_root):
            dcm_files = [f for f in filenames if f.lower().endswith(".dcm")]
            if not dcm_files:
                continue
            series_dir = Path(dirpath)
            sample_path = series_dir / dcm_files[0]
            try:
                ds = pydicom.dcmread(str(sample_path), stop_before_pixels=True)
                modality = getattr(ds, "Modality", "")
                if modality and modality.upper() == "SEG":
                    case_id = self._infer_case_id(series_dir)
                    if case_id not in catalog:
                        catalog[case_id] = {"series": {}, "default_series": ""}
                    catalog[case_id].setdefault("segmentations", []).append(str(sample_path))
                    continue
            except Exception:
                ds = None

            case_id = self._infer_case_id(series_dir)
            series_id = series_dir.name
            slice_paths = [str(series_dir / name) for name in sorted(dcm_files)]
            sop_uids = []
            for slice_path in slice_paths:
                try:
                    slice_ds = pydicom.dcmread(slice_path, stop_before_pixels=True)
                    sop_uids.append(str(getattr(slice_ds, "SOPInstanceUID", "")))
                except Exception:
                    sop_uids.append("")
            width = int(getattr(ds, "Columns", 0) or 0)
            height = int(getattr(ds, "Rows", 0) or 0)
            series_instance_uid = str(getattr(ds, "SeriesInstanceUID", ""))

            if case_id not in catalog:
                catalog[case_id] = {"series": {}, "default_series": series_id}
            elif not catalog[case_id].get("default_series"):
                catalog[case_id]["default_series"] = series_id
            catalog[case_id]["series"][series_id] = {
                "slice_paths": slice_paths,
                "sop_instance_uids": sop_uids,
                "slice_count": len(slice_paths),
                "width": width,
                "height": height,
                "series_instance_uid": series_instance_uid,
            }
        return catalog

    @staticmethod
    def _infer_case_id(path: Path) -> str:
        for parent in [path] + list(path.parents):
            name = parent.name
            if name.startswith("LIDC-IDRI"):
                return name
        return path.parent.name
