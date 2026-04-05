import csv
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom


def infer_case_id(path: Path) -> str:
    for parent in [path] + list(path.parents):
        name = parent.name
        if name.startswith("LIDC-IDRI"):
            return name
    return path.parent.name


def safe_get(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else None
    return str(value)


def extract_referenced_series(ds) -> List[str]:
    series_uids: List[str] = []
    if hasattr(ds, "ReferencedSeriesSequence"):
        for item in ds.ReferencedSeriesSequence:
            uid = getattr(item, "SeriesInstanceUID", None)
            if uid:
                series_uids.append(str(uid))
    return series_uids


def inspect_seg_file(path: Path) -> Dict:
    result: Dict = {"path": str(path), "status": "unread"}
    try:
        ds = pydicom.dcmread(str(path))
        result["status"] = "read"
        result["modality"] = safe_get(getattr(ds, "Modality", None))
        result["sop_class_uid"] = safe_get(getattr(ds, "SOPClassUID", None))
        result["series_instance_uid"] = safe_get(getattr(ds, "SeriesInstanceUID", None))
        result["study_instance_uid"] = safe_get(getattr(ds, "StudyInstanceUID", None))
        result["segment_count"] = len(getattr(ds, "SegmentSequence", []) or [])
        result["frames"] = int(getattr(ds, "NumberOfFrames", 0) or 0)
        result["referenced_series"] = extract_referenced_series(ds)
        result["referenced_sop_uids"] = extract_referenced_sop_instances(ds)
        try:
            pixels = ds.pixel_array
            result["pixel_array_shape"] = list(pixels.shape)
            bbox = compute_first_frame_bbox(pixels)
            if bbox:
                result["first_frame_bbox"] = bbox
        except Exception as exc:
            result["pixel_array_error"] = str(exc)
    except Exception as exc:
        result["error"] = str(exc)
    return result


def extract_referenced_sop_instances(ds) -> List[str]:
    sop_uids: List[str] = []
    if hasattr(ds, "PerFrameFunctionalGroupsSequence"):
        for frame in ds.PerFrameFunctionalGroupsSequence:
            derivation = getattr(frame, "DerivationImageSequence", None)
            if derivation:
                for item in derivation:
                    source_seq = getattr(item, "SourceImageSequence", None)
                    if not source_seq:
                        continue
                    for source in source_seq:
                        uid = getattr(source, "ReferencedSOPInstanceUID", None)
                        if uid:
                            sop_uids.append(str(uid))
    return sop_uids


def compute_first_frame_bbox(pixels: np.ndarray) -> Optional[Dict]:
    if pixels.ndim == 2:
        frames = [pixels]
    elif pixels.ndim == 3:
        frames = list(pixels)
    else:
        return None

    for idx, frame in enumerate(frames):
        if not np.any(frame):
            continue
        ys, xs = np.where(frame > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        height, width = frame.shape
        return {
            "frame_index": idx,
            "x": x_min,
            "y": y_min,
            "w": x_max - x_min + 1,
            "h": y_max - y_min + 1,
            "normalized": {
                "x": x_min / width,
                "y": y_min / height,
                "w": (x_max - x_min + 1) / width,
                "h": (y_max - y_min + 1) / height,
            },
        }
    return None


def main():
    root = Path("Data/medicalimages")
    report_path = Path("data/medical_report.json")
    csv_path = Path("data/medical_report.csv")

    dicom_rows: List[Dict] = []
    cases: Dict[str, Dict] = {}
    xml_files: List[str] = []
    json_files: List[str] = []
    csv_files: List[str] = []

    for path in root.rglob("*"):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        if suffix == ".xml":
            xml_files.append(str(path))
            continue
        if suffix == ".json":
            json_files.append(str(path))
            continue
        if suffix == ".csv":
            csv_files.append(str(path))
            continue
        if suffix != ".dcm":
            continue

        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:
            continue

        modality = safe_get(getattr(ds, "Modality", None))
        sop_class_uid = safe_get(getattr(ds, "SOPClassUID", None))
        series_desc = safe_get(getattr(ds, "SeriesDescription", None))
        study_uid = safe_get(getattr(ds, "StudyInstanceUID", None))
        series_uid = safe_get(getattr(ds, "SeriesInstanceUID", None))
        referenced_series = extract_referenced_series(ds)
        case_id = infer_case_id(path)

        entry = {
            "path": str(path),
            "case_id": case_id,
            "modality": modality,
            "sop_class_uid": sop_class_uid,
            "series_description": series_desc,
            "study_instance_uid": study_uid,
            "series_instance_uid": series_uid,
            "referenced_series_uids": ";".join(referenced_series),
        }
        dicom_rows.append(entry)

        cases.setdefault(case_id, {"ct": [], "seg": [], "sr": []})
        if modality == "CT":
            cases[case_id]["ct"].append(str(path))
        elif modality == "SEG":
            cases[case_id]["seg"].append(str(path))
        elif modality == "SR":
            cases[case_id]["sr"].append(str(path))

    sample_ct = next((row for row in dicom_rows if row.get("modality") == "CT"), None)
    sample_seg = next((row for row in dicom_rows if row.get("modality") == "SEG"), None)

    seg_diagnostics = inspect_seg_file(Path(sample_seg["path"])) if sample_seg else None

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "root": str(root),
        "counts": {
            "dicom_total": len(dicom_rows),
            "ct": sum(1 for row in dicom_rows if row.get("modality") == "CT"),
            "seg": sum(1 for row in dicom_rows if row.get("modality") == "SEG"),
            "sr": sum(1 for row in dicom_rows if row.get("modality") == "SR"),
            "xml": len(xml_files),
            "json": len(json_files),
            "csv": len(csv_files),
        },
        "cases": {
            case_id: {
                "ct_count": len(payload["ct"]),
                "seg_count": len(payload["seg"]),
                "sr_count": len(payload["sr"]),
                "has_ct": len(payload["ct"]) > 0,
                "has_seg": len(payload["seg"]) > 0,
                "has_sr": len(payload["sr"]) > 0,
            }
            for case_id, payload in cases.items()
        },
        "sample_ct": sample_ct,
        "sample_seg": sample_seg,
        "seg_diagnostics": seg_diagnostics,
        "xml_files": xml_files[:5],
        "json_files": json_files[:5],
        "csv_files": csv_files[:5],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "case_id",
                "modality",
                "sop_class_uid",
                "series_description",
                "study_instance_uid",
                "series_instance_uid",
                "referenced_series_uids",
            ],
        )
        writer.writeheader()
        writer.writerows(dicom_rows)

    print(f"DICOM files: {len(dicom_rows)}")
    print(f"CT: {summary['counts']['ct']}, SEG: {summary['counts']['seg']}, SR: {summary['counts']['sr']}")
    print(f"XML: {summary['counts']['xml']}, JSON: {summary['counts']['json']}, CSV: {summary['counts']['csv']}")
    print(f"Report: {report_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
